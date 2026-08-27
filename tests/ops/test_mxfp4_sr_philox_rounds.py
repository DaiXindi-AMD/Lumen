###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""The Philox round count behind stochastic-rounding dither.

Philox mixing is the largest single feature of the gradient-path quantizer, so
the round count is the one knob that trades kernel instructions against dither
quality. Cutting it too far costs nothing visible -- the kernel still runs, the
tensors still have the right shape, and training still steps -- while the dither
stops being uniform and quantization error grows.

The quantity that notices is the *spread* of the residual, not its mean: a
degraded dither is still roughly centred, so ``|mean|/std`` stays around 1e-3
whether the dither is good or ruined, and asserting on the mean would prove
nothing. Below the floor the residual std jumps several-fold and, unlike an
unbiased estimator's, stops shrinking as draws are added.
"""

import importlib
import os

import pytest
import torch

import lumen.kernels.mxfp4 as mxfp4_kernels

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

BLOCK = 32
ROUNDS_FLOOR = 4  # lowest count measured to hold dither quality
ROUNDS_BELOW_FLOOR = 2
DRAWS = 96


@pytest.fixture
def rebuild_at_rounds():
    """Rebuild the kernel module at a given round count, then put it back.

    ``SR_PHILOX_ROUNDS_C`` is a ``tl.constexpr`` closed over by the jitted
    functions, so the count has to be fixed before tracing. Assigning the
    attribute afterwards is not enough and not even allowed -- Triton 3.7
    raises "Global variable SR_PHILOX_ROUNDS_C has changed since we compiled
    this kernel" -- hence the environment variable plus a reload, which builds
    fresh JITFunctions. ``lumen.ops.quantize.ops`` imports the kernels inside
    its functions, so the next call picks up the rebuilt module.
    """
    saved = os.environ.get("LUMEN_SR_PHILOX_ROUNDS")

    def _rebuild(rounds):
        os.environ["LUMEN_SR_PHILOX_ROUNDS"] = str(rounds)
        module = importlib.reload(mxfp4_kernels)
        assert module.SR_PHILOX_ROUNDS == rounds, "reload did not take"
        return module

    yield _rebuild

    if saved is None:
        os.environ.pop("LUMEN_SR_PHILOX_ROUNDS", None)
    else:
        os.environ["LUMEN_SR_PHILOX_ROUNDS"] = saved
    importlib.reload(mxfp4_kernels)


def _dequant(packed, scales, block):
    """Packed MXFP4 + E8M0 scales -> fp32, matching the kernel's convention.

    The kernel builds the multiplier as ``(scale_byte << 23)`` bitcast to fp32,
    so the byte lands in the exponent field and the value is 2^(byte-127).
    """
    fp4_utils = pytest.importorskip("aiter.utility.fp4_utils", reason="AITER required")

    codes = fp4_utils.mxfp4_to_f32(packed)
    mult = (scales.to(torch.int32) << 23).view(torch.float32)
    return codes * mult.repeat_interleave(block, dim=-1)


def _residual_std(draws=DRAWS):
    """Spread of ``dequant(quant(x)) - x`` averaged over independent SR draws.

    Row-major layout only: the transposed operand rotates before quantizing, so
    its residual is not comparable elementwise.
    """
    from lumen.ops.quantize.ops import convert_to_mxfp4

    torch.manual_seed(0)
    shape = (256, BLOCK * 8)
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    acc = torch.zeros(shape, dtype=torch.float32, device="cuda")
    for i in range(draws):
        packed, scales = convert_to_mxfp4(
            x, BLOCK, axis=-1, use_sr=True, philox_seed=1234 + i, philox_offset=0,
        )
        acc += _dequant(packed, scales, BLOCK)
    resid = acc / draws - x.to(torch.float32)
    # Normalise by mean|x| so the figure compares across round counts.
    denom = x.to(torch.float32).abs().mean().clamp_min(1e-6)
    return (resid.std() / denom).item()


def test_default_round_count_is_the_documented_default(rebuild_at_rounds):
    """An unset environment must give the count the report's numbers came from."""
    os.environ.pop("LUMEN_SR_PHILOX_ROUNDS", None)
    module = importlib.reload(mxfp4_kernels)

    assert module.SR_PHILOX_ROUNDS == module.SR_PHILOX_ROUNDS_DEFAULT


def test_override_reaches_the_traced_constant(rebuild_at_rounds):
    """The constexpr is what the kernel closes over; the int alone is inert."""
    module = rebuild_at_rounds(ROUNDS_FLOOR)

    assert module.SR_PHILOX_ROUNDS == ROUNDS_FLOOR
    assert module.SR_PHILOX_ROUNDS_C.value == ROUNDS_FLOOR


def test_rounds_at_the_floor_hold_dither_quality(rebuild_at_rounds):
    """Four rounds is the measured floor: residual spread matches the default."""
    rebuild_at_rounds(mxfp4_kernels.SR_PHILOX_ROUNDS_DEFAULT)
    default_std = _residual_std()

    rebuild_at_rounds(ROUNDS_FLOOR)
    floor_std = _residual_std()

    # Measured at 3% apart; 15% leaves room for reduction-order variation
    # without admitting the several-fold jump seen below the floor.
    assert floor_std == pytest.approx(default_std, rel=0.15), (
        f"rounds={ROUNDS_FLOOR} residual std {floor_std:.6f} vs "
        f"default {default_std:.6f}"
    )


def test_metric_catches_a_dither_below_the_floor(rebuild_at_rounds):
    """Guards the test above: the metric must be able to fail."""
    rebuild_at_rounds(mxfp4_kernels.SR_PHILOX_ROUNDS_DEFAULT)
    default_std = _residual_std()

    rebuild_at_rounds(ROUNDS_BELOW_FLOOR)
    starved_std = _residual_std()

    assert starved_std > 2 * default_std, (
        f"rounds={ROUNDS_BELOW_FLOOR} residual std {starved_std:.6f} is not "
        f"clearly worse than default {default_std:.6f}; the metric has lost "
        "its discriminating power"
    )
