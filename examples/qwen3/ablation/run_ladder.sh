#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# Run the MXFP4 ablation ladder (docs/mxfp4_ablation_plan.md).
#
# Usage:
#   bash run_ladder.sh --dry-run --all         # print every arm's env, run nothing
#   bash run_ladder.sh S0                      # one arm
#   bash run_ladder.sh S0 S6 S8 S12 S20 S24    # the shape-confirming subset
#   bash run_ladder.sh --all                   # the whole ladder
#   REPEATS=1 bash run_ladder.sh S24           # override the repeat count
#
# Everything about the model, data and parallelism comes from
# run_pretrain_qwen3_8b_mxfp4.sh and is deliberately not settable here: arms may
# differ from each other only in ablation env.
#
# Two things this script exists to get right, both of which silently ruin the
# ladder if done by hand:
#
#   * Every arm gets its own autotune cache. The cache records which backend won
#     under whatever kernels were reachable at the time, so one shared file lets
#     an early arm pin "use Triton" for every shape and every later arm inherits
#     it -- the ladder comes out flat and nothing says why.
#   * Every arm sets every switch explicitly, including the ones it leaves off.
#     Relying on defaults would make each arm depend on what the previous run
#     happened to export.
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN3_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LUMEN_DIR="$(cd "${QWEN3_DIR}/../.." && pwd)"

# shellcheck source=arms.sh
source "${SCRIPT_DIR}/arms.sh"

MAX_ARM="$(abl_max_arm)"
DRY_RUN=0
REPEATS="${REPEATS:-2}"
TRAIN_STEPS="${TRAIN_STEPS:-60}"
SEED="${SEED:-1234}"
OUT_ROOT="${OUT_ROOT:-${QWEN3_DIR}/results/ablation}"

ARMS=()
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=1 ;;
        --all)     for n in $(seq 0 "${MAX_ARM}"); do ARMS+=("S${n}"); done ;;
        S*)        ARMS+=("${arg}") ;;
        *)         echo "unknown argument: ${arg}" >&2; exit 2 ;;
    esac
done
[ "${#ARMS[@]}" -gt 0 ] || { sed -n '8,20p' "${BASH_SOURCE[0]}"; exit 2; }

# A12 is the model's own tuned table, so its two sides are two search paths
# rather than a flag. The launcher builds the tuned list itself when the variable
# is unset; reproducing it here would drift, so the tuned side is spelled as the
# empty string and skipped, letting the launcher do it.
resolve_a4w4() {  # <tuned|untuned>
    local aiter_cfg
    aiter_cfg="$(python -c 'import aiter, os; print(os.path.join(os.path.dirname(aiter.__file__), "configs"))')"
    if [ "$1" = "tuned" ]; then
        echo ""    # unset: the launcher merges the model table in
    else
        # AITER's stock table only. The model's Qwen3-8B table is what A12 added,
        # so leaving it out is exactly this arm's "off" state.
        echo "${aiter_cfg}/a4w4_blockscale_tuned_gemm.csv"
    fi
}

A4W4_TUNED="$(resolve_a4w4 tuned)"
A4W4_UNTUNED="$(resolve_a4w4 untuned)"

run_one_arm() {  # <n> <repeat>
    local n="$1" rep="$2"
    local arm="S${n}"
    local out="${OUT_ROOT}/${arm}/run${rep}"
    local env_pairs=() kv

    while IFS= read -r kv; do
        [ -n "${kv}" ] || continue
        kv="${kv//@A4W4_TUNED@/${A4W4_TUNED}}"
        kv="${kv//@A4W4_UNTUNED@/${A4W4_UNTUNED}}"
        # An empty value means "leave it unset" (only A12's tuned side).
        [ "${kv#*=}" = "" ] && continue
        env_pairs+=("${kv}")
    done < <(abl_env_for_arm "${n}")

    echo "=============================================================="
    echo "${arm} run${rep}  $(abl_arm_summary "${n}")"
    echo "  results: ${out}"
    if [ "${DRY_RUN}" = "1" ]; then
        printf '  %s\n' "${env_pairs[@]}"
        return 0
    fi

    mkdir -p "${out}"
    printf '%s\n' "${env_pairs[@]}" > "${out}/ablation_env.txt"
    abl_arm_summary "${n}" > "${out}/arm.txt"

    env "${env_pairs[@]}" \
        RESULTS_DIR="${out}" \
        LUMEN_MXFP4_AUTOTUNE_CACHE="${out}/autotune.json" \
        TRAIN_STEPS="${TRAIN_STEPS}" \
        SEED="${SEED}" \
        bash "${QWEN3_DIR}/run_pretrain_qwen3_8b_mxfp4.sh" \
        2>&1 | tee "${out}/console.log"
}

for arm in "${ARMS[@]}"; do
    n="${arm#S}"
    if ! [ "${n}" -ge 0 ] 2>/dev/null || [ "${n}" -gt "${MAX_ARM}" ]; then
        echo "arm ${arm} out of range: the ladder is S0..S${MAX_ARM}" >&2
        exit 2
    fi
    for rep in $(seq 1 "${REPEATS}"); do
        run_one_arm "${n}" "${rep}"
    done
done

echo ""
echo "[DONE] ${#ARMS[@]} arm(s) x ${REPEATS} repeat(s) under ${OUT_ROOT}"
