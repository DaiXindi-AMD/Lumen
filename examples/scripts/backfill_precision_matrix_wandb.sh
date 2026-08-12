#!/bin/bash
###############################################################################
# Replay the BF16 / FP8 / MXFP4 precision matrix logs into W&B.
#
# The matrix was measured without --wandb-project, so Megatron never built its
# wandb writer and the metrics live only in the stdout logs. This walks those
# logs through wandb_backfill_megatron_log.py to create the equivalent runs after
# the fact. Every run lands tagged `backfilled_from_log`, and its wall-clock
# timestamps are the replay's rather than the training run's.
#
# Usage:
#   bash examples/scripts/backfill_precision_matrix_wandb.sh
#   DRY_RUN=1 bash ...            # parse and print, upload nothing
#   MODE=offline bash ...         # write locally for a later `wandb sync`
#   PROJECT=other-project bash ...
#
# MXFP4 run names carry both the GEMM table and the assembly coverage it bought,
# because that is worth 7-11% and nothing in the log or the dashboard shows it:
# `stocktable` is AITER's own table alone, `tunedtable` adds the Lumen CSV for that
# model, and `asmNof11` is how many of the 11 shapes then reached an assembly
# kernel instead of Triton. Comparing a stocktable arm against a tunedtable one
# compares two kernel sets, not two precisions
# (docs/mxfp4_precision_benchmark_report.md §7).
#
# The three stocktable arms are one condition, not two: Llama2-7B and Llama3.1-8B
# reached it by being measured before their CSV existed, Qwen3-8B by having its
# removed. Their coverage differs (2, 4 and 3 of 11) only because AITER's stock
# table happens to cover each model's shapes differently, so the coverage belongs
# in the name and the condition does not need two names for it.
###############################################################################
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUMEN_DIR="${LUMEN_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RESULTS="${LUMEN_DIR}/examples"
BACKFILL="${LUMEN_DIR}/examples/qwen3/scripts/wandb_backfill_megatron_log.py"

PROJECT="${PROJECT:-lumen-precision-matrix-mi350x}"
MODE="${MODE:-online}"

# name <TAB> log <TAB> baseline log ("-" for the BF16 arms, which are the baseline)
#
# Every cell ran at SEED=1234 over the same mock corpus, so any BF16 run of the
# same model is a paired baseline: at a given iteration both arms have seen the
# same batches, which is what makes delta/lm_loss_vs_baseline a comparison rather
# than two unrelated curves. Repeats are paired with their own repeat so the
# BF16-vs-BF16 row shows the noise floor as a curve.
read -r -d '' RUNS <<'EOF'
llama2_7b-bf16	llama2/results/lumen_llama2_7b_bf16.log	-
llama2_7b-bf16-rep2	llama2/results/lumen_llama2_7b_rep2_bf16.log	llama2/results/lumen_llama2_7b_bf16.log
llama2_7b-fp8-e4m3	llama2/results/lumen_llama2_7b_fp8.log	llama2/results/lumen_llama2_7b_bf16.log
llama2_7b-fp8-e4m3-nativelinear	llama2/results/lumen_llama2_7b_lumlin_fp8.log	llama2/results/lumen_llama2_7b_bf16.log
llama2_7b-mxfp4-stocktable-asm2of11	llama2/results/lumen_llama2_7b_mxfp4.log	llama2/results/lumen_llama2_7b_bf16.log
llama2_7b-mxfp4-stocktable-asm2of11-rep2	llama2/results/lumen_llama2_7b_rep2_mxfp4.log	llama2/results/lumen_llama2_7b_rep2_bf16.log
llama2_7b-mxfp4-tunedtable-asm11of11	llama2/results/lumen_llama2_7b_tunedgemm_mxfp4.log	llama2/results/lumen_llama2_7b_bf16.log
llama31_8b-bf16	llama31/results/lumen_llama31_8b_bf16.log	-
llama31_8b-bf16-rep2	llama31/results/lumen_llama31_8b_rep2_bf16.log	llama31/results/lumen_llama31_8b_bf16.log
llama31_8b-fp8-e4m3	llama31/results/lumen_llama31_8b_fp8.log	llama31/results/lumen_llama31_8b_bf16.log
llama31_8b-mxfp4-stocktable-asm4of11	llama31/results/lumen_llama31_8b_mxfp4.log	llama31/results/lumen_llama31_8b_bf16.log
llama31_8b-mxfp4-stocktable-asm4of11-rep2	llama31/results/lumen_llama31_8b_rep2_mxfp4.log	llama31/results/lumen_llama31_8b_rep2_bf16.log
llama31_8b-mxfp4-tunedtable-asm11of11	llama31/results/lumen_llama31_8b_tunedgemm_mxfp4.log	llama31/results/lumen_llama31_8b_bf16.log
qwen3_8b-bf16	qwen3/results/lumen_qwen3_8b_bf16.log	-
qwen3_8b-bf16-rep2	qwen3/results/lumen_qwen3_8b_rep2_bf16.log	qwen3/results/lumen_qwen3_8b_bf16.log
qwen3_8b-fp8-e4m3	qwen3/results/lumen_qwen3_8b_fp8.log	qwen3/results/lumen_qwen3_8b_bf16.log
qwen3_8b-fp8-e4m3-nativelinear	qwen3/results/lumen_qwen3_8b_lumlin_fp8.log	qwen3/results/lumen_qwen3_8b_bf16.log
qwen3_8b-mxfp4-tunedtable-asm11of11	qwen3/results/lumen_qwen3_8b_mxfp4.log	qwen3/results/lumen_qwen3_8b_bf16.log
qwen3_8b-mxfp4-tunedtable-asm11of11-rep2	qwen3/results/lumen_qwen3_8b_rep2_mxfp4.log	qwen3/results/lumen_qwen3_8b_rep2_bf16.log
qwen3_8b-mxfp4-stocktable-asm3of11	qwen3/results/lumen_qwen3_8b_stocktable_mxfp4.log	qwen3/results/lumen_qwen3_8b_bf16.log
EOF

ok=0
failed=0
while IFS=$'\t' read -r name log baseline; do
    [ -n "${name}" ] || continue
    log="${RESULTS}/${log}"
    if [ ! -s "${log}" ]; then
        echo "[skip] ${name}: no log at ${log}"
        failed=$((failed + 1))
        continue
    fi

    args=("${log}" --project "${PROJECT}" --name "${name}" --mode "${MODE}")
    [ "${baseline}" = "-" ] || args+=(--baseline-log "${RESULTS}/${baseline}")
    [ "${DRY_RUN:-0}" = "1" ] && args+=(--dry-run)
    [ -n "${ENTITY:-}" ] && args+=(--entity "${ENTITY}")

    echo ""
    echo "=== ${name}"
    if python "${BACKFILL}" "${args[@]}"; then
        ok=$((ok + 1))
    else
        echo "[FAILED] ${name}"
        failed=$((failed + 1))
    fi
done <<< "${RUNS}"

echo ""
echo "[backfill] ${ok} ok, ${failed} failed, project=${PROJECT} mode=${MODE}"
