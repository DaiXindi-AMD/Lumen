"""The staircase's ablation switches have to be faithful and complete.

Two properties matter. Every switch defaults to the optimized path, so a run
with no environment set reproduces HEAD -- if that breaks, every arm shifts at
once. And a switch that claims to remove something must actually remove it:
``LUMEN_MXFP4_ASM`` / ``LUMEN_MXFP4_PRESHUFFLE`` only steer the static byte
thresholds, so autotune still measures and picks those backends behind them.
Reusing them as ablation switches would report "removing ASM costs nothing".

See docs/mxfp4_ablation_plan.md sections 4.2 and 7.
"""
import pytest
import torch

from lumen.ops.quantize import mxfp4_autotune
from lumen.utils import ablation


_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _is_gfx950():
    if not torch.cuda.is_available():
        return False
    return "gfx950" in torch.cuda.get_device_properties(0).gcnArchName


_GFX950 = pytest.mark.skipif(not _is_gfx950(), reason="gfx950 operand layout")


@pytest.fixture(autouse=True)
def _clean_switch_cache():
    ablation.reset()
    yield
    ablation.reset()


def test_every_switch_defaults_to_the_optimized_path(monkeypatch):
    for name in ablation._REGISTRY:
        monkeypatch.delenv(f"LUMEN_ABL_{name}", raising=False)
    ablation.reset()
    assert ablation.active_overrides() == []


def test_switch_reads_the_environment(monkeypatch):
    monkeypatch.setenv("LUMEN_ABL_VEC_SHUFFLE", "0")
    ablation.reset()
    assert not ablation.enabled("VEC_SHUFFLE")
    assert ablation.active_overrides() == ["VEC_SHUFFLE"]


def test_unknown_switch_raises():
    # A mistyped switch that silently did nothing would produce an arm that
    # measures the wrong optimization, which is worse than a crash.
    with pytest.raises(KeyError):
        ablation.enabled("VEC_SHUFFEL")


def test_every_switch_names_the_commit_it_reverts():
    for name in ablation._REGISTRY:
        commit, what = ablation.describe(name)
        assert len(commit) >= 7 and what


def _choose_with_stub(monkeypatch, cached_name, *, gates):
    """``_mxfp4_choose_backend`` with the probes forced on and autotune stubbed.

    Returns ``(chosen, candidate_names)``. The operands only need shapes: every
    gated path short-circuits before anything reads their storage.
    """
    import lumen.ops.quantize.linear as lin

    monkeypatch.setattr(lin, "_fast_mxfp4_gemm_probed", True)
    monkeypatch.setattr(lin, "_fast_mxfp4_gemm_fn", lambda *a: None)
    monkeypatch.setattr(lin, "_fast_mxfp4_asm_ok", True)
    monkeypatch.setattr(lin, "_fast_mxfp4_preshuffle_ok", True)
    monkeypatch.setattr(mxfp4_autotune, "cached", lambda key: cached_name)
    monkeypatch.setattr(mxfp4_autotune, "record_shape", lambda key, **kw: None)

    seen = []

    def _pick(key, candidates, fallback=None):
        seen.extend(name for name, _ in candidates)
        return fallback

    monkeypatch.setattr(mxfp4_autotune, "pick_backend", _pick)

    a_fp4 = torch.empty(512, 256, dtype=torch.uint8)
    w_fp4 = torch.empty(1024, 256, dtype=torch.uint8)
    scale = torch.empty(1, 1, dtype=torch.uint8)
    with ablation.overridden(**gates):
        chosen = lin._mxfp4_choose_backend(a_fp4, w_fp4, scale, scale)
    return chosen, seen


def test_asm_switch_takes_asm_out_of_the_candidate_list(monkeypatch):
    _, seen = _choose_with_stub(
        monkeypatch, None,
        gates={"MXFP4_ASM_BACKEND": False, "MXFP4_SHUF_BACKEND": True},
    )
    assert "asm" not in seen


def test_shuffled_switch_takes_shuffled_out_of_the_candidate_list(monkeypatch):
    _, seen = _choose_with_stub(
        monkeypatch, None,
        gates={"MXFP4_ASM_BACKEND": True, "MXFP4_SHUF_BACKEND": False},
    )
    assert "shuffled" not in seen


def test_both_switches_off_leaves_only_the_plain_kernel(monkeypatch):
    chosen, seen = _choose_with_stub(
        monkeypatch, None,
        gates={"MXFP4_ASM_BACKEND": False, "MXFP4_SHUF_BACKEND": False},
    )
    assert seen == ["plain"] and chosen == "plain"


@pytest.mark.parametrize("stale", ["asm", "shuffled"])
def test_switch_overrides_a_stale_autotune_decision(monkeypatch, stale):
    """An autotune cache recorded while the backend was reachable must not win.

    The cache records whatever kernels the run that wrote it could reach, so a
    shared cache file would otherwise re-enable the very backend this arm is
    measuring the absence of.
    """
    chosen, _ = _choose_with_stub(
        monkeypatch, stale,
        gates={"MXFP4_ASM_BACKEND": False, "MXFP4_SHUF_BACKEND": False},
    )
    assert chosen == "plain"


def test_stale_autotune_decision_is_honoured_when_the_switch_is_on(monkeypatch):
    chosen, seen = _choose_with_stub(
        monkeypatch, "asm",
        gates={"MXFP4_ASM_BACKEND": True, "MXFP4_SHUF_BACKEND": True},
    )
    assert chosen == "asm" and seen == []


@_CUDA
@_GFX950
def test_vec_shuffle_switch_is_bit_exact():
    """The wide shuffle only changes how many bytes move per element."""
    import lumen.ops.quantize.linear as lin

    # Above _MXFP4_WIDE_SHUFFLE_MIN_BYTES, or the wide path declines anyway.
    w = torch.randint(
        0, 255, (2048, 4096), dtype=torch.uint8, device="cuda",
    )
    with ablation.overridden(VEC_SHUFFLE=True):
        wide = lin._shuffle_mxfp4_weight(w, arch="gfx950")
    with ablation.overridden(VEC_SHUFFLE=False):
        narrow = lin._shuffle_mxfp4_weight(w, arch="gfx950")
    torch.testing.assert_close(wide, narrow, atol=0, rtol=0)


@_CUDA
@_GFX950
def test_scale_pad_skip_switch_is_bit_exact():
    """Skipping the copy is only legal because the copy was byte-identical."""
    import lumen.ops.quantize.linear as lin
    from lumen.ops.quantize.ops import triton_arch

    tiling = lin._MXFP4_SCALE_SHUFFLE_TILING.get(triton_arch())
    if tiling is None:
        pytest.skip("no scale shuffle tiling for this arch")

    rows = lin._MXFP4_ASM_SCALE_ROW_MULTIPLE * 4
    cols = lin._MXFP4_ASM_SCALE_COL_MULTIPLE * 4
    scale = torch.randint(0, 255, (rows, cols), dtype=torch.uint8, device="cuda")

    with ablation.overridden(SWIZZLE_CACHE=True, SCALE_PAD_SKIP=True):
        skipped = lin._pad_and_swizzle_mxfp4_scale(scale, triton_arch(), tiling)
    with ablation.overridden(SWIZZLE_CACHE=True, SCALE_PAD_SKIP=False):
        copied = lin._pad_and_swizzle_mxfp4_scale(scale, triton_arch(), tiling)
    torch.testing.assert_close(skipped, copied, atol=0, rtol=0)


@_CUDA
@_GFX950
def test_mfma_h16_switch_routes_the_rotation_off_the_matrix_unit():
    """Only the MFMA path reads the folded matrix; the butterfly reads the signs.

    A zeroed matrix therefore has to blank the output with the switch on and be
    inert with it off. Without this the bit-exactness test below would also pass
    on a switch that was never wired to anything.
    """
    import lumen.ops.quantize.linear as lin
    from lumen.ops.quantize.ops import (
        _RHT_MATRIX_ATTR, _rht_matrix_bf16, hadamard_quant_mxfp4,
    )

    torch.manual_seed(3)
    x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    sign = lin._get_mxfp4_rht_sign(x.device)
    g = lin._MXFP4_RHT_G

    def _quant():
        return hadamard_quant_mxfp4(x, sign, block_size=32, g=g, use_sr=False)[0]

    base = {}
    for on in (True, False):
        with ablation.overridden(MFMA_H16=on):
            base[on] = _quant()

    good = _rht_matrix_bf16(sign, g).clone()
    setattr(sign, _RHT_MATRIX_ATTR, torch.zeros_like(good))
    try:
        with ablation.overridden(MFMA_H16=True):
            assert bool((_quant() == 0).all()), "the matrix unit did not read hmat"
        with ablation.overridden(MFMA_H16=False):
            torch.testing.assert_close(_quant(), base[False], atol=0, rtol=0)
    finally:
        setattr(sign, _RHT_MATRIX_ATTR, good)


@_CUDA
@_GFX950
def test_mfma_h16_switch_is_bit_exact():
    """Both forms sum the same 16 exact products, and E2M1 is coarse enough.

    The orders differ, so the FP32 sums may differ in their last bits, but the
    two-mantissa-bit output grid does not resolve that -- measured bit-equal over
    shapes, seeds and a 1e4 row-magnitude spread.
    """
    import lumen.ops.quantize.linear as lin
    from lumen.ops.quantize.ops import hadamard_quant_mxfp4

    sign = lin._get_mxfp4_rht_sign(torch.device("cuda"))
    g = lin._MXFP4_RHT_G
    for seed in (0, 42):
        for shape in ((256, 256), (512, 1024)):
            for spread in (1.0, 1e4):
                torch.manual_seed(seed)
                x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
                x = x * (spread ** torch.rand(shape[0], 1, device="cuda")).bfloat16()
                got = {}
                for on in (True, False):
                    with ablation.overridden(MFMA_H16=on):
                        got[on] = hadamard_quant_mxfp4(
                            x, sign, block_size=32, g=g, use_sr=False,
                        )
                for a, b in zip(got[True], got[False]):
                    torch.testing.assert_close(
                        a, b, atol=0, rtol=0,
                        msg=f"seed={seed} shape={shape} spread={spread:g}",
                    )


def _run_linear(**switches):
    """One MXFP4 linear forward+backward under *switches*, seeds pinned.

    The philox counters come from Python's RNG, so the seed has to be reset for
    each run or two arms that draw the same number of times still see different
    streams and nothing is comparable.
    """
    import random

    from lumen.ops.quantize.linear import QuantizedLinearFunction

    torch.manual_seed(5)
    x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)

    random.seed(0)
    torch.manual_seed(0)
    xi = x.detach().clone().requires_grad_(True)
    wi = w.detach().clone().requires_grad_(True)
    with ablation.overridden(**switches):
        QuantizedLinearFunction.apply(
            xi, wi, None, None, "mxfp4", None, 32, "weight",
        ).backward(dy)
    return xi.grad, wi.grad


# A2, A5 and A16 sit on the activation's WGrad operand, which forward supplies
# outright once A18 is on, and A2 also sits on the gradient operand, which the
# dual-layout kernel supplies once A19 is on. Turning any of them off alone at
# HEAD therefore changes nothing -- their code is only reached in the early rungs,
# where A18 and A19 are still off. Tests have to use that base or they pass
# vacuously.
_EARLY_RUNG = {"FWD_WGRAD_OPERAND": False, "DUAL_LAYOUT": False}


@_CUDA
@_GFX950
@pytest.mark.parametrize(
    "switch,base",
    [
        ("DGRAD_WEIGHT_REUSE", {}),
        ("WGRAD_VIEWS", {}),
        ("FUSED_DHQ", _EARLY_RUNG),
        ("DEQUANT_TRANSPOSE", _EARLY_RUNG),
    ],
)
def test_backward_switch_is_bit_exact(switch, base):
    """Reusing the weight, passing views, and both fusions keep dX and dW exact."""
    ref_dx, ref_dw = _run_linear(**base)
    got_dx, got_dw = _run_linear(**{**base, switch: False})
    torch.testing.assert_close(got_dx, ref_dx, atol=0, rtol=0, msg=f"{switch} moved dX")
    torch.testing.assert_close(got_dw, ref_dw, atol=0, rtol=0, msg=f"{switch} moved dW")


@_CUDA
@_GFX950
def test_fused_hq_wgrad_is_not_bit_exact_but_is_equivalent():
    """The unfused rotation narrows to BF16 in between, so dW moves a little.

    ``hadamard_transform`` returns the input dtype, so the two-kernel form writes
    a BF16 intermediate that the fused kernel keeps in FP32. That is a real
    numerical difference rather than a re-draw, but it is far below the noise the
    MXFP4 gradient already carries.
    """
    ref_dx, ref_dw = _run_linear(**_EARLY_RUNG)
    got_dx, got_dw = _run_linear(**_EARLY_RUNG, FUSED_HQ_WGRAD=False)

    torch.testing.assert_close(got_dx, ref_dx, atol=0, rtol=0, msg="dX should not move")
    assert not torch.equal(got_dw, ref_dw), (
        "expected the BF16 intermediate to change dW; if this now matches, the "
        "unfused path stopped narrowing and the switch may be unwired"
    )
    assert _snr(ref_dw, got_dw) > 30.0, (
        f"the unfused rotation should differ only marginally, got "
        f"{_snr(ref_dw, got_dw):.2f} dB against the fused result"
    )


@_CUDA
@_GFX950
def test_fwd_wgrad_operand_improves_dw():
    """A18's rung is expected to move the loss, and in the good direction.

    Forward's operand skips one quantization round, so dW lands closer to the
    BF16 reference. The plan reports that as a remark on the optimization, which
    only holds if the gain is real and sizeable.
    """
    torch.manual_seed(5)
    x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
    exact_dw = (dy.float().t() @ x.float()).bfloat16()

    _, fused_dw = _run_linear()
    _, rebuilt_dw = _run_linear(FWD_WGRAD_OPERAND=False, FUSED_DHQ=False)
    assert _snr(exact_dw, fused_dw) > _snr(exact_dw, rebuilt_dw) + 0.5, (
        f"expected forward's operand to be the better dW: "
        f"{_snr(exact_dw, fused_dw):.2f} vs {_snr(exact_dw, rebuilt_dw):.2f} dB"
    )


@_CUDA
@_GFX950
def test_rtn_skip_philox_only_changes_the_random_stream():
    """The RTN kernels never read the counter, so skipping the draw is free.

    What the draw costs is the stream: every SR caller after it sees a different
    one, which is why this arm moves the loss without changing any single RTN
    result.
    """
    import random

    from lumen.ops.quantize.ops import convert_to_mxfp4

    torch.manual_seed(3)
    x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)

    out, states = {}, {}
    for on in (True, False):
        random.seed(0)
        with ablation.overridden(RTN_SKIP_PHILOX=on):
            out[on] = convert_to_mxfp4(x, block_size=32, axis=-1, use_sr=False)
        states[on] = random.random()

    for a, b in zip(out[True], out[False]):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert states[True] != states[False], (
        "the stream did not move, so the switch never reached the draw"
    )


@_CUDA
@_GFX950
def test_every_switch_off_still_runs():
    """S0 is a combination that never existed; it has to at least execute.

    Each switch was verified alone, but the stripped baseline turns all of them
    off at once, and the legacy paths compose in ways no single-switch test
    reaches.
    """
    dx, dw = _run_linear(**{name: False for name in ablation._REGISTRY})
    for name, g in (("dX", dx), ("dW", dw)):
        assert g is not None and torch.isfinite(g).all(), f"{name} is not finite at S0"


def _snr(ref, got):
    err = (ref.float() - got.float()).pow(2).sum()
    if err == 0:
        return float("inf")
    return 10 * torch.log10(ref.float().pow(2).sum() / err).item()


@_CUDA
@_GFX950
def test_dual_layout_switch_is_bit_exact_without_stochastic_rounding():
    """One read or two, the arithmetic is the same; only the reads differ."""
    import lumen.ops.quantize.linear as lin
    from lumen.ops.quantize.ops import (
        convert_to_mxfp4, dual_layout_quant_mxfp4, hadamard_quant_mxfp4,
    )

    torch.manual_seed(3)
    x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    sign = lin._get_mxfp4_rht_sign(x.device)
    g = lin._MXFP4_RHT_G

    row, row_s, col, col_s = dual_layout_quant_mxfp4(
        x, sign, block_size=32, g=g, use_sr_row=False, use_sr_transposed=False,
    )
    row2, row_s2 = convert_to_mxfp4(x, block_size=32, axis=-1, use_sr=False)
    col2, col_s2 = hadamard_quant_mxfp4(
        x.t(), sign, block_size=32, g=g, use_sr=False,
    )
    for got, want in ((row, row2), (row_s, row_s2), (col, col2), (col_s, col_s2)):
        torch.testing.assert_close(got, want, atol=0, rtol=0)


@_CUDA
@_GFX950
def test_dual_layout_switch_only_redraws_stochastic_rounding():
    """Under SR the two forms differ, and the staircase has to expect that.

    Backward quantizes the gradient with SR on, and the fused kernel maps philox
    counters to elements over its own tiling, so the same seed lands different
    draws. The block scales still come out bit-identical and neither form is
    closer to the exact rotation, so this is a re-draw and not a quality change
    -- the A19 rung may move the loss the way a different seed would.
    """
    import lumen.ops.quantize.linear as lin
    from lumen.ops.quantize.ops import (
        convert_from_mxfp4, convert_to_mxfp4, dual_layout_quant_mxfp4,
        hadamard_quant_mxfp4, hadamard_transform,
    )

    torch.manual_seed(3)
    x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    sign = lin._get_mxfp4_rht_sign(x.device)
    g = lin._MXFP4_RHT_G
    seed, offset = 7, 13

    row, row_s, col, col_s = dual_layout_quant_mxfp4(
        x, sign, block_size=32, g=g, use_sr_row=True, use_sr_transposed=True,
        philox_seed=seed, philox_offset=offset,
    )
    row2, row_s2 = convert_to_mxfp4(
        x, block_size=32, axis=-1, use_sr=True,
        philox_seed=seed, philox_offset=offset,
    )
    col2, col_s2 = hadamard_quant_mxfp4(
        x.t(), sign, block_size=32, g=g, use_sr=True,
        philox_seed=seed, philox_offset=offset,
    )

    torch.testing.assert_close(row_s, row_s2, atol=0, rtol=0, msg="row scales moved")
    torch.testing.assert_close(col_s, col_s2, atol=0, rtol=0, msg="col scales moved")

    def _dq(data, scale):
        return convert_from_mxfp4(
            data, scale, output_dtype=torch.float32, block_size=32,
        )

    row_ref = x.float()
    col_ref = hadamard_transform(x.t().contiguous(), sign, g=g).float()
    for ref, fused, split in (
        (row_ref, _dq(row, row_s), _dq(row2, row_s2)),
        (col_ref, _dq(col, col_s), _dq(col2, col_s2)),
    ):
        assert abs(_snr(ref, fused) - _snr(ref, split)) < 0.5, (
            f"one form rounds better: {_snr(ref, fused):.2f} dB vs {_snr(ref, split):.2f} dB"
        )


@_CUDA
@_GFX950
def test_dual_layout_switch_routes_backward_to_the_two_call_form():
    """The switch has to trade exactly one fused call for one two-call pair.

    ``dual_layout_quant_mxfp4`` serves two optimizations: forward's WGrad
    activation operand (A18) and backward's gradient (A19). Only the second is
    this switch's, so the count drops by one rather than to zero.
    """
    import lumen.ops.quantize.ops as ops
    from lumen.ops.quantize.linear import QuantizedLinearFunction

    fused, split = ops.dual_layout_quant_mxfp4, ops.hadamard_quant_mxfp4

    torch.manual_seed(5)
    x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)

    def _count(dual_layout):
        calls = {"fused": 0, "split": 0}

        def _fused(*a, **kw):
            calls["fused"] += 1
            return fused(*a, **kw)

        def _split(*a, **kw):
            calls["split"] += 1
            return split(*a, **kw)

        monkey = pytest.MonkeyPatch()
        try:
            monkey.setattr(ops, "dual_layout_quant_mxfp4", _fused)
            monkey.setattr(ops, "hadamard_quant_mxfp4", _split)
            xi = x.detach().clone().requires_grad_(True)
            wi = w.detach().clone().requires_grad_(True)
            with ablation.overridden(DUAL_LAYOUT=dual_layout):
                QuantizedLinearFunction.apply(
                    xi, wi, None, None, "mxfp4", None, 32, "weight",
                ).backward(dy)
            assert wi.grad is not None
        finally:
            monkey.undo()
        return calls

    try:
        on = _count(True)
        off = _count(False)
    except RuntimeError as e:
        pytest.skip(f"Lumen MXFP4 path unavailable: {e}")

    assert on["fused"] >= 1, "the fused form was not used by default"
    assert off["fused"] == on["fused"] - 1, (
        f"expected one fewer fused call, got {off['fused']} vs {on['fused']}"
    )
    assert off["split"] == on["split"] + 1, (
        f"expected one more two-call pair, got {off['split']} vs {on['split']}"
    )


@_CUDA
@_GFX950
def test_swizzle_cache_switch_rebuilds_the_same_operands():
    import lumen.ops.quantize.linear as lin

    w = torch.randint(0, 255, (256, 128), dtype=torch.uint8, device="cuda")
    scale = torch.randint(0, 255, (256, 8), dtype=torch.uint8, device="cuda")
    calls = []

    def _build():
        calls.append(1)
        return (w.clone(), scale.clone())

    with ablation.overridden(SWIZZLE_CACHE=True):
        first = lin._cached_weight_operands(w, scale, "_test_operands", _build)
        second = lin._cached_weight_operands(w, scale, "_test_operands", _build)
    assert second is first and len(calls) == 1

    with ablation.overridden(SWIZZLE_CACHE=False):
        third = lin._cached_weight_operands(w, scale, "_test_operands", _build)
    assert len(calls) == 2
    for got, want in zip(third, first):
        torch.testing.assert_close(got, want, atol=0, rtol=0)
