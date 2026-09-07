"""SFT Benchmark — plan.md §8.2 evaluation.

Generates FlyDSL kernel code from instructions at 5 difficulty levels,
then evaluates: API usage, code structure, Python syntax, pattern matching.
Compares base model vs SFT model.

Usage::

    python eval_sft.py \
        --base-model /dev/shm/qwen2.5-coder-32b \
        --sft-model /home/danyzhan/sft-results/Qwen2.5-Coder-SFT \
        --output /home/danyzhan/sft-results/benchmark.json
"""

import argparse
import json
import logging
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a FlyDSL GPU kernel programming expert. You write compilable, "
    "correct, high-performance GPU kernels using the FlyDSL framework for "
    "AMD Instinct GPUs. Always use @flyc.kernel decorator, fx.* expression "
    "API, and follow FlyDSL coding patterns."
)

# 25 test prompts across 5 difficulty levels (plan.md §8.2)
TEST_PROMPTS = [
    # Level 1 (beginner): vec_add, relu, reduction
    {"level": 1, "id": "L1_vec_add", "prompt": "Write a FlyDSL kernel that performs element-wise vector addition C = A + B for two float32 vectors of length N. Use @flyc.kernel decorator and fx.gpu thread indexing."},
    {"level": 1, "id": "L1_relu", "prompt": "Write a FlyDSL kernel that applies ReLU activation (max(0, x)) element-wise to a BF16 tensor. Use @flyc.kernel and fx.gpu.thread_idx for parallelization."},
    {"level": 1, "id": "L1_scale", "prompt": "Write a FlyDSL kernel that scales a vector by a scalar constant: Y = alpha * X. Use @flyc.kernel decorator with proper thread indexing."},
    {"level": 1, "id": "L1_copy", "prompt": "Write a FlyDSL kernel that copies data from one buffer to another using buffer_ops.buffer_load and buffer_ops.buffer_store."},
    {"level": 1, "id": "L1_reduce", "prompt": "Write a FlyDSL kernel that computes the sum reduction of a float32 vector using shared memory and fx.syncthreads."},
    # Level 2 (basic): softmax, RMSNorm, LayerNorm
    {"level": 2, "id": "L2_softmax", "prompt": "Write a FlyDSL kernel implementing row-wise softmax for a 2D BF16 tensor. Handle numerical stability with max subtraction. Use @flyc.kernel and fx.* APIs."},
    {"level": 2, "id": "L2_rmsnorm", "prompt": "Write a FlyDSL kernel implementing RMSNorm normalization for a hidden state tensor. Compute RMS = sqrt(mean(x^2) + eps), then normalize and scale by weight parameter."},
    {"level": 2, "id": "L2_layernorm", "prompt": "Write a FlyDSL kernel implementing LayerNorm with learnable gamma and beta parameters. Use shared memory for mean/variance computation."},
    {"level": 2, "id": "L2_silu", "prompt": "Write a FlyDSL kernel implementing SiLU (Swish) activation: y = x * sigmoid(x) for BF16 tensors. Use @flyc.kernel decorator."},
    {"level": 2, "id": "L2_rope", "prompt": "Write a FlyDSL kernel implementing Rotary Position Embedding (RoPE) for query/key tensors with cos/sin cached frequencies."},
    # Level 3 (intermediate): simple GEMM, TopK, fused ops
    {"level": 3, "id": "L3_gemm_naive", "prompt": "Write a FlyDSL GEMM kernel computing C = A @ B for BF16 matrices. Use tiled computation with fx.make_layout, SmemAllocator for shared memory, and MFMA instructions via rocdl.mfma_f32_32x32x16_bf16."},
    {"level": 3, "id": "L3_topk", "prompt": "Write a FlyDSL kernel that finds the top-K elements and their indices from each row of a 2D tensor. Use shared memory for partial sorting."},
    {"level": 3, "id": "L3_fused_bias_relu", "prompt": "Write a FlyDSL kernel that fuses bias addition and ReLU activation: Y = max(0, X + bias). Optimize by avoiding extra memory traffic with kernel fusion."},
    {"level": 3, "id": "L3_gemv", "prompt": "Write a FlyDSL kernel for matrix-vector multiplication Y = A @ x where A is (M,K) BF16 and x is (K,) BF16. Use fx.make_layout for tiling."},
    {"level": 3, "id": "L3_concat", "prompt": "Write a FlyDSL kernel that concatenates two tensors along a given dimension with proper layout handling using fx.make_layout and fx.make_shape."},
    # Level 4 (advanced): FP8 GEMM, FlashAttn, PagedAttn
    {"level": 4, "id": "L4_fp8_gemm", "prompt": "Write a FlyDSL FP8 GEMM kernel for gfx950 with 2-stage pipeline, swizzle_xor16 for LDS bank conflict avoidance, and preshuffle data layout. Use @flyc.kernel with SmemAllocator and lds_stage=2."},
    {"level": 4, "id": "L4_flash_attn", "prompt": "Write a FlyDSL FlashAttention forward kernel for BF16 on gfx950. Implement online softmax with running max/sum, tiled Q@K^T and score@V computation using MFMA instructions and shared memory."},
    {"level": 4, "id": "L4_paged_attn", "prompt": "Write a FlyDSL PagedAttention decode kernel for KV-cache serving with block_table lookup, supporting variable-length sequences and page_size=16."},
    {"level": 4, "id": "L4_gemm_splitk", "prompt": "Write a FlyDSL split-K GEMM kernel that partitions the K dimension across workgroups and uses an atomic reduction to accumulate partial results. Target gfx942 with BF16."},
    {"level": 4, "id": "L4_fused_norm_quant", "prompt": "Write a FlyDSL kernel that fuses RMSNorm + FP8 quantization in a single pass. Compute norm, then quantize the normalized output to fp8_e4m3fnuz with per-tensor scaling."},
    # Level 5 (expert): Preshuffle GEMM, MoE, MLA
    {"level": 5, "id": "L5_preshuffle_gemm", "prompt": "Write a FlyDSL preshuffle GEMM kernel for gfx950 FP8 with tile 128x128x256, double-buffered LDS pipeline (lds_stage=2), swizzle_xor16, 4 waves per workgroup, and epilogue fusion for bias+activation. Use SmemAllocator, fx.zipped_divide, fx.composition for layout algebra."},
    {"level": 5, "id": "L5_moe_2stage", "prompt": "Write a FlyDSL 2-stage MoE GEMM kernel supporting top-K expert routing with token permutation. Stage 1 computes gate_proj+up_proj, stage 2 computes down_proj. Use pipeline for LDS double buffering and expert-parallel tiling."},
    {"level": 5, "id": "L5_mla_decode", "prompt": "Write a FlyDSL MLA (Multi-head Latent Attention) decode kernel for inference. Support compressed KV with latent_dim=576, nope/rope split, and head_dim=128. Use MFMA instructions for BF16 matmul."},
    {"level": 5, "id": "L5_blockscale_gemm", "prompt": "Write a FlyDSL blockscale preshuffle GEMM kernel for gfx950 with per-block FP8 scaling (block_size=128), preshuffle data layout, 3-stage pipeline, and swizzle_xor16. Include proper scale factor broadcasting in the epilogue."},
    {"level": 5, "id": "L5_allreduce", "prompt": "Write a FlyDSL custom all-reduce kernel for multi-GPU ring reduction using DPP (Data Parallel Primitives) instructions. Support BF16 with warp-level shuffles and cross-workgroup synchronization."},
]

# FlyDSL API patterns for static analysis
FLYDSL_PATTERNS = {
    "flyc_kernel": (r"@flyc\.kernel|@flyc\.jit", "Has @flyc.kernel/@flyc.jit decorator"),
    "fx_api": (r"fx\.\w+", "Uses fx.* expression API"),
    "fx_gpu": (r"fx\.gpu\.", "Uses fx.gpu thread/block indexing"),
    "fx_layout": (r"fx\.make_layout|fx\.make_shape|fx\.make_stride", "Uses fx layout algebra"),
    "smem_alloc": (r"SmemAllocator|smem_alloc", "Uses SmemAllocator for shared memory"),
    "buffer_ops": (r"buffer_ops\.|buffer_load|buffer_store", "Uses buffer_ops for memory access"),
    "mfma": (r"rocdl\.mfma|mfma_", "Uses MFMA matrix instructions"),
    "syncthreads": (r"fx\.syncthreads|syncthreads", "Uses thread synchronization"),
    "swizzle": (r"swizzle_xor|swizzle", "Uses swizzle for bank conflict avoidance"),
    "pipeline": (r"lds_stage|pipeline|num_stages", "Uses pipeline/LDS staging"),
    "zipped_divide": (r"fx\.zipped_divide", "Uses fx.zipped_divide"),
    "composition": (r"fx\.composition", "Uses fx.composition"),
    "import_flyc": (r"import flydsl|from flydsl|import flyc|from flyc", "Imports FlyDSL modules"),
    "valid_python": (None, "Valid Python syntax"),
}

LEVEL_EXPECTED_PATTERNS = {
    1: ["flyc_kernel", "fx_api", "import_flyc"],
    2: ["flyc_kernel", "fx_api", "import_flyc"],
    3: ["flyc_kernel", "fx_api", "fx_layout", "smem_alloc", "import_flyc"],
    4: ["flyc_kernel", "fx_api", "fx_layout", "smem_alloc", "mfma", "pipeline", "import_flyc"],
    5: ["flyc_kernel", "fx_api", "fx_layout", "smem_alloc", "mfma", "pipeline", "swizzle", "import_flyc"],
}


def generate_response(model, tokenizer, prompt, device="cuda", max_new_tokens=2048):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response


def analyze_code(code):
    """Static analysis of generated FlyDSL code."""
    results = {}
    for name, (pattern, desc) in FLYDSL_PATTERNS.items():
        if name == "valid_python":
            try:
                compile(code, "<gen>", "exec")
                results[name] = True
            except SyntaxError:
                results[name] = False
        else:
            results[name] = bool(re.search(pattern, code))
    return results


def score_response(analysis, level):
    """Score a response based on expected patterns for its level."""
    expected = LEVEL_EXPECTED_PATTERNS.get(level, [])
    if not expected:
        return 0.0
    matched = sum(1 for p in expected if analysis.get(p, False))
    return matched / len(expected)


def evaluate_model(model_path, model_name, tokenizer, device="cuda"):
    logger.info("=== Evaluating: %s ===", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map=device,
    )

    results = []
    for test in TEST_PROMPTS:
        logger.info("  [L%d] %s ...", test["level"], test["id"])
        response = generate_response(model, tokenizer, test["prompt"], device)
        analysis = analyze_code(response)
        score = score_response(analysis, test["level"])

        results.append({
            "id": test["id"],
            "level": test["level"],
            "prompt": test["prompt"][:100],
            "response_len": len(response),
            "analysis": analysis,
            "api_score": score,
            "response_preview": response[:500],
        })
        logger.info("    score=%.2f | patterns: %s",
                     score, {k: v for k, v in analysis.items() if v})

    del model
    torch.cuda.empty_cache()
    return results


def print_report(base_results, sft_results):
    print("\n" + "=" * 80)
    print("  SFT BENCHMARK REPORT (plan.md §8.2)")
    print("=" * 80)

    for level in range(1, 6):
        level_names = {1: "beginner", 2: "basic", 3: "intermediate", 4: "advanced", 5: "expert"}
        base_level = [r for r in base_results if r["level"] == level]
        sft_level = [r for r in sft_results if r["level"] == level]

        if not base_level:
            continue

        print(f"\n--- Level {level} ({level_names[level]}) ---")
        print(f"{'Test':<22} {'Base Score':>12} {'SFT Score':>12} {'Δ':>8}")
        print("-" * 56)

        for b, s in zip(base_level, sft_level):
            delta = s["api_score"] - b["api_score"]
            d_str = f"+{delta:.0%}" if delta > 0 else f"{delta:.0%}"
            print(f"{b['id']:<22} {b['api_score']:>11.0%} {s['api_score']:>11.0%} {d_str:>8}")

        b_avg = sum(r["api_score"] for r in base_level) / len(base_level)
        s_avg = sum(r["api_score"] for r in sft_level) / len(sft_level)
        print(f"{'Level avg':<22} {b_avg:>11.0%} {s_avg:>11.0%} {s_avg-b_avg:>+7.0%}")

    # Overall summary
    print(f"\n{'=' * 80}")
    print(f"  OVERALL SUMMARY")
    print(f"{'=' * 80}")

    for name, key in [
        ("@flyc.kernel usage", "flyc_kernel"),
        ("fx.* API usage", "fx_api"),
        ("fx.make_layout", "fx_layout"),
        ("SmemAllocator", "smem_alloc"),
        ("MFMA instructions", "mfma"),
        ("Pipeline/LDS staging", "pipeline"),
        ("Swizzle", "swizzle"),
        ("import flydsl", "import_flyc"),
        ("Valid Python syntax", "valid_python"),
    ]:
        b_pct = sum(1 for r in base_results if r["analysis"].get(key)) / len(base_results) * 100
        s_pct = sum(1 for r in sft_results if r["analysis"].get(key)) / len(sft_results) * 100
        print(f"  {name:<25} Base: {b_pct:5.1f}%   SFT: {s_pct:5.1f}%   Δ: {s_pct-b_pct:+5.1f}%")

    b_total = sum(r["api_score"] for r in base_results) / len(base_results)
    s_total = sum(r["api_score"] for r in sft_results) / len(sft_results)
    print(f"\n  {'Overall API Score':<25} Base: {b_total:5.1%}   SFT: {s_total:5.1%}   Δ: {s_total-b_total:+5.1%}")

    # Per-level summary
    print(f"\n  Per-Level API Score:")
    targets = {1: 0.9, 2: 0.85, 3: 0.7, 4: 0.5, 5: 0.2}
    for level in range(1, 6):
        s_level = [r for r in sft_results if r["level"] == level]
        avg = sum(r["api_score"] for r in s_level) / len(s_level) if s_level else 0
        target = targets[level]
        status = "PASS" if avg >= target else "FAIL"
        print(f"    Level {level}: {avg:5.1%} (target: {target:.0%}) [{status}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--sft-model", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    base_results = evaluate_model(args.base_model, "Qwen2.5-Coder-32B (base)", tokenizer, args.device)
    sft_results = evaluate_model(args.sft_model, "Qwen2.5-Coder-SFT", tokenizer, args.device)

    print_report(base_results, sft_results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"base": base_results, "sft": sft_results}, f, indent=2, default=str)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
