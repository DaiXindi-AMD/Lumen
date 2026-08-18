###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# The MXFP4 ablation ladder, in the order the optimizations were introduced.
#
# One source of truth for the arm definitions, sourced by run_ladder.sh and
# checked by tests/ops/test_ablation_ladder_arms.py. Design and per-item
# reasoning: docs/mxfp4_ablation_plan.md.
#
# The ladder is cumulative: arm Sn has A1..An on and the rest off, so S0 is the
# stripped baseline and S24 must equal HEAD with no ablation env at all. Order is
# the commit timestamp, not the date -- three of these landed within four minutes
# of each other on 07-29 and the wrong order makes two rungs read as zero.
#
# Each entry is "Aid|label|env when enabled|env when disabled". Most arms are one
# LUMEN_ABL_* flag; the ones that already had a runtime switch use it instead of
# growing a second way to say the same thing. Two are inverted or indirect and
# are called out where they appear.
###############################################################################

# @A4W4_TUNED@ / @A4W4_UNTUNED@ are resolved by run_ladder.sh: the tuned side has
# to be the launcher's own list (it merges AITER's stock table with the model's),
# and the untuned side is that same list minus the model's table.
ABL_LADDER=(
    "A1|DGrad reuses forward's transposed FP4 weight|LUMEN_ABL_DGRAD_WEIGHT_REUSE=1|LUMEN_ABL_DGRAD_WEIGHT_REUSE=0"
    "A2|WGrad rotates and quantizes in one kernel|LUMEN_ABL_FUSED_HQ_WGRAD=1|LUMEN_ABL_FUSED_HQ_WGRAD=0"
    "A3|fast GEMM dispatch|LUMEN_FAST_QUANT_DISPATCH=1|LUMEN_FAST_QUANT_DISPATCH=0"
    # Inverted: the existing switch names the disable, not the feature.
    "A4|weight cache across micro-batches|LUMEN_MXFP4_DISABLE_WEIGHT_CACHE=0|LUMEN_MXFP4_DISABLE_WEIGHT_CACHE=1"
    "A5|fused dequant+transpose|LUMEN_ABL_DEQUANT_TRANSPOSE=1|LUMEN_ABL_DEQUANT_TRANSPOSE=0"
    "A6|preshuffle backend in dispatch|LUMEN_ABL_MXFP4_SHUF_BACKEND=1|LUMEN_ABL_MXFP4_SHUF_BACKEND=0"
    "A7|ASM backend in dispatch|LUMEN_ABL_MXFP4_ASM_BACKEND=1|LUMEN_ABL_MXFP4_ASM_BACKEND=0"
    "A8|measured autotune over static thresholds|LUMEN_MXFP4_AUTOTUNE=1|LUMEN_MXFP4_AUTOTUNE=0"
    "A9|skip redundant scale padding|LUMEN_ABL_SCALE_PAD_SKIP=1|LUMEN_ABL_SCALE_PAD_SKIP=0"
    "A10|vectorised wide weight shuffle|LUMEN_ABL_VEC_SHUFFLE=1|LUMEN_ABL_VEC_SHUFFLE=0"
    "A11|WGrad passes transposed views|LUMEN_ABL_WGRAD_VIEWS=1|LUMEN_ABL_WGRAD_VIEWS=0"
    # Indirect: the arm is the model's own tuned table, not a feature flag.
    "A12|Qwen3-8B tuned A4W4 table|AITER_CONFIG_GEMM_A4W4=@A4W4_TUNED@|AITER_CONFIG_GEMM_A4W4=@A4W4_UNTUNED@"
    "A13|fused RoPE|FUSED_ROPE=1|FUSED_ROPE=0"
    "A14|no philox draw on the RTN paths|LUMEN_ABL_RTN_SKIP_PHILOX=1|LUMEN_ABL_RTN_SKIP_PHILOX=0"
    "A15|H16 rotation on the matrix unit|LUMEN_ABL_MFMA_H16=1|LUMEN_ABL_MFMA_H16=0"
    "A16|fused dequant+Hadamard+quant|LUMEN_ABL_FUSED_DHQ=1|LUMEN_ABL_FUSED_DHQ=0"
    "A17|cached/fused scale swizzle|LUMEN_ABL_SWIZZLE_CACHE=1|LUMEN_ABL_SWIZZLE_CACHE=0"
    "A18|forward emits the WGrad activation operand|LUMEN_ABL_FWD_WGRAD_OPERAND=1|LUMEN_ABL_FWD_WGRAD_OPERAND=0"
    "A19|dual-layout gradient quantization|LUMEN_ABL_DUAL_LAYOUT=1|LUMEN_ABL_DUAL_LAYOUT=0"
    "A20|quantizer emits the shuffled B operand|LUMEN_ABL_QUANT_EMIT_SHUFFLE=1|LUMEN_ABL_QUANT_EMIT_SHUFFLE=0"
    "A21|narrow-N RMSNorm backward|LUMEN_ABL_NARROW_N_RMSNORM=1|LUMEN_ABL_NARROW_N_RMSNORM=0"
    "A22|attention reads strided QKV views|LUMEN_ABL_ATTN_QKV_VIEWS=1|LUMEN_ABL_ATTN_QKV_VIEWS=0"
    "A23|seq-major attention output|LUMEN_ABL_ATTN_SEQ_MAJOR=1|LUMEN_ABL_ATTN_SEQ_MAJOR=0"
    "A24|gc.freeze|LUMEN_GC_FREEZE=1|LUMEN_GC_FREEZE=0"
)

# Highest arm index. S0 is the baseline, so there are this many arms plus one.
abl_max_arm() { echo "${#ABL_LADDER[@]}"; }

abl_arm_field() {  # <A-index 1-based> <field 1-4>
    echo "${ABL_LADDER[$(($1 - 1))]}" | cut -d'|' -f"$2"
}

# The env for arm S<n>: every optimization up to n enabled, the rest disabled.
abl_env_for_arm() {  # <n>
    local n="$1" i
    for i in $(seq 1 "${#ABL_LADDER[@]}"); do
        if [ "$i" -le "$n" ]; then
            abl_arm_field "$i" 3
        else
            abl_arm_field "$i" 4
        fi
    done
}

# What this arm turns on relative to the previous one, for logs and reports.
abl_arm_summary() {  # <n>
    local n="$1"
    if [ "$n" = "0" ]; then
        echo "stripped baseline, every optimization off"
    else
        echo "+ $(abl_arm_field "$n" 1) $(abl_arm_field "$n" 2)"
    fi
}
