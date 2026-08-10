#!/usr/bin/env bash
# Run the MXFP4 test suite and print one status line per group.
#
# Groups are run as separate pytest processes on purpose: an abort inside one
# AITER JIT op used to take the whole session with it, hiding every result
# behind it. Logs land in $OUT for the failures that need reading.
#
# Results belong in docs/mxfp4_test_report.md -- that file is the single place
# MXFP4 test status is recorded.
set -u

OUT=${OUT:-/tmp/mxfp4_tests}
MEGATRON_ROOT=${MEGATRON_ROOT:-/home/xdai/Megatron-LM}
export PYTHONPATH=${MEGATRON_ROOT}${PYTHONPATH:+:$PYTHONPATH}

mkdir -p "$OUT"
cd "$(dirname "$0")/.."

fail=0

run() {
  local name=$1; shift
  timeout "${TIMEOUT:-2400}" python -m pytest "$@" -q --no-header -p no:cacheprovider \
    > "$OUT/$name.log" 2>&1
  local code=$?
  local summary
  summary=$(rg '(passed|failed|error)( |,|$)' "$OUT/$name.log" | tail -1)
  printf '%-24s exit=%-3s %s\n' "$name" "$code" "$summary"
  [ "$code" -ne 0 ] && fail=1
  return 0
}

echo "== MXFP4 test suite =="
python -c "import torch; p = torch.cuda.get_device_properties(0); \
print(f'{torch.cuda.device_count()} x {p.gcnArchName}')" 2>/dev/null

run quantize_mxfp4      tests/ops/test_quantize.py -k "mxfp4 or hadamard or unaligned or dividing"
run linear_mxfp4        tests/ops/test_linear.py -k mxfp4
run dual_layout         tests/ops/test_mxfp4_dual_layout_shuffle.py
run fused_act_scale     tests/ops/test_mxfp4_fused_act_scale.py
run fwd_wgrad_operand   tests/ops/test_mxfp4_fwd_wgrad_operand.py
run weight_cache_hook   tests/quantize/test_mxfp4_weight_cache_hook.py
run weight_operand_fuse tests/quantize/test_mxfp4_weight_operand_fusion.py
run backward_opt        tests/test_mxfp4_backward_optimization.py
run grad_quant          tests/core/test_grad_quant.py
run models_mxfp4        tests/models/test_megatron.py tests/models/test_fsdp.py \
                        tests/models/test_megatron_entrypoint_parity.py -k mxfp4

echo
echo "logs: $OUT"
exit $fail
