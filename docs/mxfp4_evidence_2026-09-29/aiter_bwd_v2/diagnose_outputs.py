import hashlib

import torch

from aiter.ops.triton.activation import swiglu_bwd_split
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.quant import dual_layout_quant_mxfp4 as aiter_dual
from aiter.ops.triton.quant import (
    fused_swiglu_bwd_dual_layout_mxfp4 as aiter_fused_bwd,
)
from lumen.ops.quantize.ops import dual_layout_quant_mxfp4 as lumen_dual

BLOCK_SIZE = 32
HADAMARD_SIZE = 16


def digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def snr(reference: torch.Tensor, actual: torch.Tensor) -> float:
    ref = reference.float()
    error = ref - actual.float()
    signal = torch.linalg.vector_norm(ref)
    noise = torch.linalg.vector_norm(error)
    if float(noise) == 0.0:
        return float("inf")
    return float(20.0 * torch.log10(signal / noise))


def compare(label: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    difference = actual != expected
    count = int(difference.sum())
    if actual.is_floating_point():
        max_abs = float((actual.float() - expected.float()).abs().max())
    else:
        max_abs = int((actual.to(torch.int32) - expected.to(torch.int32)).abs().max())
    first = None
    if count:
        index = difference.nonzero()[0]
        first_tuple = tuple(int(value) for value in index)
        first = (
            first_tuple,
            (
                float(actual[first_tuple])
                if actual.is_floating_point()
                else int(actual[first_tuple])
            ),
            (
                float(expected[first_tuple])
                if expected.is_floating_point()
                else int(expected[first_tuple])
            ),
        )
    print(
        f"{label}: equal={count == 0} mismatch={count}/{actual.numel()} "
        f"max_abs={max_abs} first={first} actual_sha={digest(actual)} "
        f"expected_sha={digest(expected)}"
    )


def normalized_hadamard16(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    matrix = torch.ones((1, 1), dtype=torch.float32, device=device)
    while matrix.shape[0] < HADAMARD_SIZE:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return (matrix * 0.25).to(dtype)


def h16_transposed_fp32(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    blocks = x.T.reshape(cols, rows // HADAMARD_SIZE, HADAMARD_SIZE)
    matrix = normalized_hadamard16(x.device, torch.float32)
    return (blocks.float() @ matrix).reshape(cols, rows)


def h16_transposed_bf16_output(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    blocks = x.T.reshape(cols, rows // HADAMARD_SIZE, HADAMARD_SIZE)
    matrix = normalized_hadamard16(x.device, torch.bfloat16)
    return (blocks @ matrix).reshape(cols, rows)


def dequantize(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    packed_u8 = packed.view(torch.uint8)
    codes = torch.empty(
        (*packed_u8.shape[:-1], packed_u8.shape[-1] * 2),
        dtype=torch.uint8,
        device=packed.device,
    )
    codes[..., 0::2] = packed_u8 & 0x0F
    codes[..., 1::2] = packed_u8 >> 4
    values = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=packed.device,
    )[codes.long()]
    scale_values = torch.ldexp(
        torch.ones(scales.shape, dtype=torch.float32, device=scales.device),
        scales.to(torch.int32) - 127,
    )
    return values * scale_values.repeat_interleave(BLOCK_SIZE, dim=-1)


def diagnose_dual(rows: int, cols: int = 256) -> None:
    print(f"CASE dual_rtn_{rows}x{cols}")
    torch.manual_seed(20260928 + rows)
    x = torch.randn((rows, cols), dtype=torch.bfloat16, device="cuda") * 2
    sign = torch.ones(HADAMARD_SIZE, dtype=torch.float32, device="cuda")
    aiter = aiter_dual(x, use_sr=False)
    lumen = lumen_dual(
        x,
        sign,
        block_size=BLOCK_SIZE,
        g=HADAMARD_SIZE,
        use_sr_row=False,
        use_sr_transposed=False,
    )
    row_ref = dynamic_mxfp4_quant(x)
    fp32_rot = h16_transposed_fp32(x)
    bf16_rot = h16_transposed_bf16_output(x)
    fp32_col_ref = dynamic_mxfp4_quant(fp32_rot)
    bf16_col_ref = dynamic_mxfp4_quant(bf16_rot)
    names = ("row", "row_scale", "col", "col_scale")
    for name, actual, expected in zip(names, aiter, lumen):
        compare(f"aiter_vs_lumen.{name}", actual, expected)
    compare("aiter_vs_fp32_math.row", aiter[0], row_ref[0])
    compare("aiter_vs_fp32_math.row_scale", aiter[1], row_ref[1])
    compare("aiter_vs_fp32_math.col", aiter[2], fp32_col_ref[0])
    compare("aiter_vs_fp32_math.col_scale", aiter[3], fp32_col_ref[1])
    compare("aiter_vs_bf16_output.col", aiter[2], bf16_col_ref[0])
    compare("aiter_vs_bf16_output.col_scale", aiter[3], bf16_col_ref[1])
    print(
        "quality.col_vs_fp32="
        f"{snr(fp32_rot, dequantize(aiter[2], aiter[3])):.6f}dB "
        "col_vs_bf16_output="
        f"{snr(bf16_rot, dequantize(aiter[2], aiter[3])):.6f}dB"
    )


def diagnose_bwd(rows: int, use_sr: bool) -> None:
    print(f"CASE bwd_{'sr' if use_sr else 'rtn'}_{rows}x256")
    torch.manual_seed(11000 + rows)
    grad = torch.randn((rows, 256), dtype=torch.bfloat16, device="cuda")
    gate = torch.randn_like(grad)
    up = torch.randn_like(grad)
    dgate, dup = swiglu_bwd_split(grad, gate, up)
    sign = torch.ones(HADAMARD_SIZE, dtype=torch.float32, device="cuda")
    fused_kwargs = {"use_sr": use_sr}
    dual_kwargs = {"use_sr": use_sr}
    if use_sr:
        fused_kwargs.update(
            gate_philox_seed=101,
            gate_philox_offset=1009,
            up_philox_seed=211,
            up_philox_offset=2003,
        )
        gate_dual_kwargs = {**dual_kwargs, "philox_seed": 101, "philox_offset": 1009}
        up_dual_kwargs = {**dual_kwargs, "philox_seed": 211, "philox_offset": 2003}
    else:
        gate_dual_kwargs = dual_kwargs
        up_dual_kwargs = dual_kwargs
    fused = aiter_fused_bwd(grad, gate, up, **fused_kwargs)
    split_aiter = (
        dgate,
        dup,
        *aiter_dual(dgate, **gate_dual_kwargs),
        *aiter_dual(dup, **up_dual_kwargs),
    )
    dgate_lumen = lumen_dual(
        dgate,
        sign,
        use_sr_row=use_sr,
        use_sr_transposed=use_sr,
        philox_seed=101 if use_sr else None,
        philox_offset=1009 if use_sr else None,
    )
    dup_lumen = lumen_dual(
        dup,
        sign,
        use_sr_row=use_sr,
        use_sr_transposed=use_sr,
        philox_seed=211 if use_sr else None,
        philox_offset=2003 if use_sr else None,
    )
    split_lumen = (dgate, dup, *dgate_lumen, *dup_lumen)
    names = (
        "dgate",
        "dup",
        "dgate_row",
        "dgate_row_scale",
        "dgate_col",
        "dgate_col_scale",
        "dup_row",
        "dup_row_scale",
        "dup_col",
        "dup_col_scale",
    )
    for name, actual, expected in zip(names, fused, split_aiter):
        compare(f"fused_vs_aiter.{name}", actual, expected)
    for name, actual, expected in zip(names, fused, split_lumen):
        compare(f"fused_vs_lumen.{name}", actual, expected)
    for name, actual, expected in zip(names, split_aiter, split_lumen):
        compare(f"aiter_vs_lumen.{name}", actual, expected)


def main() -> None:
    assert torch.cuda.is_available()
    print(f"torch={torch.__version__}")
    print(f"device={torch.cuda.get_device_name(0)}")
    for rows in (32, 64, 96, 128, 256):
        diagnose_dual(rows)
    for use_sr in (False, True):
        for rows in (32, 64, 96, 128, 256):
            diagnose_bwd(rows, use_sr)
    torch.cuda.synchronize()
    print("DIAGNOSTIC_COMPLETE")


if __name__ == "__main__":
    main()
