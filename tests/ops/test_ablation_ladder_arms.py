###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""The ladder definition has to agree with the switch registry.

``examples/qwen3/ablation/arms.sh`` and ``lumen/utils/ablation.py`` are edited for
different reasons and drift apart quietly: a switch added to the code but not the
ladder is simply never ablated, and a switch misspelled in the ladder is exported
into the environment where nothing reads it. Either way every arm still runs and
the report comes out looking finished, so the mismatch has to be a test failure
rather than something to notice in a log.
"""

import pathlib
import subprocess

import pytest

from lumen.utils import ablation

_ARMS_SH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "examples" / "qwen3" / "ablation" / "arms.sh"
)


def _arm_env(n):
    """The env pairs bash resolves for arm S<n>, as a dict."""
    out = subprocess.run(
        ["bash", "-c", f'source "{_ARMS_SH}"; abl_env_for_arm {n}'],
        capture_output=True, text=True, check=True,
    ).stdout
    pairs = {}
    for line in out.splitlines():
        if line.strip():
            key, _, value = line.partition("=")
            pairs[key] = value
    return pairs


def _max_arm():
    out = subprocess.run(
        ["bash", "-c", f'source "{_ARMS_SH}"; abl_max_arm'],
        capture_output=True, text=True, check=True,
    ).stdout
    return int(out.strip())


def test_arms_file_exists():
    assert _ARMS_SH.is_file(), f"no ladder definition at {_ARMS_SH}"


def test_every_registered_switch_is_ablated_exactly_once():
    text = _ARMS_SH.read_text()
    for name in ablation._REGISTRY:
        flag = f"LUMEN_ABL_{name}"
        on = text.count(f"{flag}=1")
        off = text.count(f"{flag}=0")
        assert on == 1 and off == 1, (
            f"{flag} appears {on}x enabled / {off}x disabled in the ladder; "
            f"every registered switch needs exactly one rung"
        )


def test_ladder_names_no_switch_the_code_does_not_have():
    import re

    text = _ARMS_SH.read_text()
    named = {m.group(1) for m in re.finditer(r"LUMEN_ABL_([A-Z0-9_]+)=", text)}
    unknown = named - set(ablation._REGISTRY)
    assert not unknown, (
        f"the ladder sets {sorted(unknown)}, which no switch reads; a typo here "
        f"produces an arm that measures nothing"
    )


def test_baseline_arm_disables_everything():
    env = _arm_env(0)
    for name in ablation._REGISTRY:
        assert env[f"LUMEN_ABL_{name}"] == "0", f"S0 left {name} on"
    # The pre-existing runtime switches, whose polarity is easy to invert.
    assert env["LUMEN_FAST_QUANT_DISPATCH"] == "0"
    assert env["LUMEN_MXFP4_AUTOTUNE"] == "0"
    assert env["LUMEN_GC_FREEZE"] == "0"
    assert env["FUSED_ROPE"] == "0"
    assert env["LUMEN_MXFP4_DISABLE_WEIGHT_CACHE"] == "1", (
        "this one names the disable, so the baseline has to set it to 1"
    )


def test_top_arm_matches_head_defaults():
    """The last arm must ask for exactly what an unset environment already does."""
    env = _arm_env(_max_arm())
    for name in ablation._REGISTRY:
        assert env[f"LUMEN_ABL_{name}"] == "1", f"the top arm left {name} off"
        assert ablation.enabled(name), f"{name} does not default to on in the code"
    assert env["LUMEN_MXFP4_DISABLE_WEIGHT_CACHE"] == "0"
    assert env["LUMEN_FAST_QUANT_DISPATCH"] == "1"


def test_ladder_is_cumulative():
    """Each arm may only add to the previous one, never take something away."""
    max_arm = _max_arm()
    previous = _arm_env(0)
    for n in range(1, max_arm + 1):
        current = _arm_env(n)
        changed = {k for k in current if current[k] != previous.get(k)}
        assert len(changed) == 1, (
            f"S{n} changes {sorted(changed)} relative to S{n - 1}; a cumulative "
            f"rung has to move exactly one thing or its delta is unattributable"
        )
        previous = current


@pytest.mark.parametrize("n", [0, 1, 12, 24])
def test_arm_env_covers_every_knob(n):
    """No arm may leave a knob unset and inherit whatever ran before it."""
    env = _arm_env(n)
    expected = len(ablation._REGISTRY) + 6  # + the six pre-existing switches
    assert len(env) == expected, (
        f"S{n} sets {len(env)} variables, expected {expected}: "
        f"{sorted(env)}"
    )
