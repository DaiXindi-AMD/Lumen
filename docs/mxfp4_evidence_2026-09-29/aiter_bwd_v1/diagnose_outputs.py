import hashlib

import torch

from aiter.ops.triton.activation import swiglu_bwd_split
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.quant import dual_layout_quant_mxfp4 as aiter_dual
from aiter.ops.triton.quant import (
    fused_swiglu_bwd_dual_layout_mxfp4 as aiter_fused_bwd,
)
from lumen.ops.quantize.ops import dual_layout_quant_mxfp4 as lumen_dual


def digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def compare(label: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    difference = actual != expected
    count = int(difference.sum())
    max_abs = int((actual.to(torch.int16) - expected.to(torch.int16)).abs().max())
    first = None
    if count:
        index = difference.nonzero()[0]
        first_tuple = tuple(int(value) for value in index)
        first = (first_tuple, int(actual[first_tuple]), int(expected[first_tuple]))
    print(
        f"{label}: equal={count == 0} mismatch={count}/{actual.numel()} "
        f"max_abs={max_abs} first={first} actual_sha={digest(actual)} "
        f"expected_sha={digest(expected)}"
    )


def h16_transposed(x: torch.Tensor) -> torch.Tensor:
    matrix = torch.ones((1, 1), dtype=torch.float32, device=x.device)
    while matrix.shape[0] < 16:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    matrix *= 0.25
    return (x.T.reshape(x.shape[1], x.shape[0] // 16, 16).float() @ matrix).reshape(
        x.shape[1], x.shape[0]
    )


def diagnose_rtn() -> None:
    print("CASE rtn_256x256")
    torch.manual_seed(20260928)
    x = torch.randn((256, 256), dtype=torch.bfloat16, device="cuda") * 2
    sign = torch.ones(16, dtype=torch.float32, device="cuda")
    aiter = aiter_dual(x, use_sr=False)
    lumen = lumen_dual(
        x,
        sign,
        block_size=32,
        g=16,
        use_sr_row=False,
        use_sr_transposed=False,
    )
    row_ref = dynamic_mxfp4_quant(x)
    col_ref = dynamic_mxfp4_quant(h16_transposed(x))
    reference = (*row_ref, *col_ref)
    names = ("row", "row_scale", "col", "col_scale")
    for name, actual, expected in zip(names, aiter, reference):
        compare(f"aiter_vs_math.{name}", actual, expected)
    for name, actual, expected in zip(names, lumen, reference):
        compare(f"lumen_vs_math.{name}", actual, expected)
    for name, actual, expected in zip(names, aiter, lumen):
        compare(f"aiter_vs_lumen.{name}", actual, expected)


def diagnose_bwd(rows: int) -> None:
    print(f"CASE bwd_{rows}x256")
    torch.manual_seed(11)
    grad = torch.randn((rows, 256), dtype=torch.bfloat16, device="cuda")
    gate = torch.randn_like(grad)
    up = torch.randn_like(grad)
    dgate, dup = swiglu_bwd_split(grad, gate, up)
    sign = torch.ones(16, dtype=torch.float32, device="cuda")
    fused = aiter_fused_bwd(
        grad,
        gate,
        up,
        use_sr=True,
        gate_philox_seed=101,
        gate_philox_offset=1009,
        up_philox_seed=211,
        up_philox_offset=2003,
    )
    split_aiter = (
        dgate,
        dup,
        *aiter_dual(dgate, use_sr=True, philox_seed=101, philox_offset=1009),
        *aiter_dual(dup, use_sr=True, philox_seed=211, philox_offset=2003),
    )
    dgate_lumen = lumen_dual(
        dgate,
        sign,
        use_sr_row=True,
        use_sr_transposed=True,
        philox_seed=101,
        philox_offset=1009,
    )
    dup_lumen = lumen_dual(
        dup,
        sign,
        use_sr_row=True,
        use_sr_transposed=True,
        philox_seed=211,
        philox_offset=2003,
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
    print(f"device={torch.cuda.get_device_name(0)}")
    diagnose_rtn()
    diagnose_bwd(64)
    diagnose_bwd(256)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
