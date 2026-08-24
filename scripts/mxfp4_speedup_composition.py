"""Why does MXFP4's speedup differ across the three benchmark models?

MXFP4 only accelerates the transformer linears, and only those outside the BF16
tail. Attention, the norms, and the LM head stay BF16 in every arm, so the
achievable speedup is an Amdahl ceiling set by how much of the step is
quantizable -- which is a property of the architecture, not of the kernels.

This computes that composition from the launcher configs and compares the
resulting ceiling against the measured speedups in
docs/mxfp4_precision_benchmark_report.md.
"""

# Backward costs two GEMMs (dgrad + wgrad) per forward one. Flash attention
# recomputes the score matrix in backward, so it runs nearer 2.5x forward.
GEMM_BWD_MULT = 3.0
ATTN_BWD_MULT = 3.5


class Model:
    def __init__(self, name, layers, hidden, ffn, heads, kv_heads, head_dim,
                 vocab, mbs, seq, qk_layernorm, bf16_tail, bf16_ms, mxfp4_ms):
        self.__dict__.update(locals())
        del self.self

    @property
    def tokens(self):
        return self.mbs * self.seq

    @property
    def attn_linear_params(self):
        q = self.hidden * self.heads * self.head_dim
        kv = 2 * self.hidden * self.kv_heads * self.head_dim
        o = self.heads * self.head_dim * self.hidden
        return q + kv + o

    @property
    def mlp_params(self):
        return 3 * self.hidden * self.ffn

    def flops(self):
        """Forward FLOPs per micro-batch, split by what MXFP4 can accelerate."""
        per_layer = 2 * self.tokens * (self.attn_linear_params + self.mlp_params)
        quantized_layers = self.layers - self.bf16_tail

        # Causal attention: QK^T and AV, each 2*seq^2*head_dim per head, halved
        # by the mask. Per sequence, so multiply by mbs rather than tokens.
        attn_per_layer = 2 * self.seq * self.seq * self.head_dim * self.heads * self.mbs
        lm_head = 2 * self.tokens * self.hidden * self.vocab

        # QK-LayerNorm normalizes q and k per head; bandwidth-bound, but count
        # its elements so the asymmetry is visible rather than hidden.
        qkn = 0
        if self.qk_layernorm:
            elems = self.tokens * (self.heads + self.kv_heads) * self.head_dim
            qkn = 4 * elems * self.layers

        return {
            "linear_quantized": per_layer * quantized_layers * GEMM_BWD_MULT,
            "linear_bf16_tail": per_layer * self.bf16_tail * GEMM_BWD_MULT,
            "attention": attn_per_layer * self.layers * ATTN_BWD_MULT,
            "lm_head": lm_head * GEMM_BWD_MULT,
            "qk_layernorm": qkn,
        }

    def params_b(self):
        body = (self.attn_linear_params + self.mlp_params) * self.layers
        return (body + 2 * self.vocab * self.hidden) / 1e9


MODELS = [
    Model("Llama2-7B", 32, 4096, 11008, 32, 32, 128, 32000, 4, 4096,
          False, 5, 6475.1, 4247.9),
    Model("Llama3.1-8B", 32, 4096, 14336, 32, 8, 128, 128256, 2, 8192,
          False, 5, 8209.5, 5720.7),
    Model("Qwen3-8B", 36, 4096, 12288, 32, 8, 128, 151936, 2, 8192,
          True, 5, 8573.6, 6097.2),
]

print(f"{'model':<14}{'params':>8}{'quantizable':>13}{'attn':>8}{'lm_head':>9}"
      f"{'bf16 tail':>11}{'measured':>10}{'ceiling':>9}{'of ceiling':>12}")
print("-" * 94)

for m in MODELS:
    f = m.flops()
    total = sum(f.values())
    p = f["linear_quantized"] / total
    measured = m.bf16_ms / m.mxfp4_ms
    # Amdahl ceiling if the quantized GEMMs became free.
    ceiling = 1.0 / (1.0 - p)
    print(f"{m.name:<14}{m.params_b():>7.2f}B{p:>12.1%}"
          f"{f['attention']/total:>8.1%}{f['lm_head']/total:>9.1%}"
          f"{f['linear_bf16_tail']/total:>11.1%}"
          f"{measured:>9.3f}x{ceiling:>8.2f}x{measured/ceiling:>11.1%}")

print()
print("Implied MXFP4 GEMM speedup needed to reach the measured step time,")
print("solving  1/((1-p) + p/s) = measured  for s:")
for m in MODELS:
    f = m.flops()
    p = f["linear_quantized"] / sum(f.values())
    measured = m.bf16_ms / m.mxfp4_ms
    s = p / (1.0 / measured - (1.0 - p))
    print(f"  {m.name:<14} p={p:.3f}  s={s:.2f}x")

print()
print("Qwen3 vs Llama3.1 -- same seq, same token count, near-identical linear")
print("FLOPs. Where the quantizable fraction goes instead:")
q, l = MODELS[2], MODELS[1]
fq, fl = q.flops(), l.flops()
for k in ("attention", "lm_head", "qk_layernorm"):
    dq = fq[k] / sum(fq.values())
    dl = fl[k] / sum(fl.values())
    print(f"  {k:<16} Qwen3 {dq:>6.1%}   Llama3.1 {dl:>6.1%}   delta {dq-dl:+.1%}")
