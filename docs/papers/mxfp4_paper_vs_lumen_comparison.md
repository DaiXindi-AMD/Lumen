# 论文对比：《Pretraining Large Language Models with MXFP4 on Native FP4 Hardware》 vs Lumen MXFP4 链路

- 论文：arXiv:2605.09825 (Cim, Palangappa, Hodak, Dwivedula, Arunachalam, Kandemir — Penn State + AMD)
- PDF：`docs/papers/mxfp4_native_fp4_hardware_2605.09825.pdf`
- Lumen 侧参照：`docs/mxfp4_status_report.md`、`docs/mxfp4_training_report.md`、`docs/mxfp4_debug_flow.md`、`lumen/ops/quantize/linear.py`、`lumen/ops/quantize/ops.py`

> 本文档反映**当前实现状态**。论文对 Lumen 的核心指导（Wgrad 是主因、确定性 Hadamard、H16）均已落地并验证：8B MXFP4 已稳定收敛 5000 步。第三条稳定手段"末尾层保 BF16"来自 NVFP4 论文而非本文，见 §3(3)。历史排查过程见 `docs/mxfp4_debug_flow.md`。

---

## 1. 论文核心结论（一句话版）

在 Llama-3.1-8B 全量 MXFP4 预训练中，**Wgrad（权重梯度）量化是收敛变差的主因**；随机化手段（Stochastic Rounding、随机 Hadamard 旋转）**在 Wgrad 也被量化后完全无法收敛**；只有**确定性 Hadamard 旋转（无随机符号翻转）**才能稳定训练，同时把 token 开销压到 8–9%（相比 FP8），端到端提速 9–10%。

论文的诊断链路：

| 阶段递进 | Token 开销（相对 FP8）|
|---|---|
| Fprop only (MXFP4) | 8–9% |
| Fprop + Dgrad (MXFP4) | 10–11% |
| Fprop + Dgrad + Wgrad (MXFP4, 无稳定手段) | **26–27%** |
| Fprop+Dgrad+Wgrad + Stochastic Rounding | **不收敛** |
| Fprop+Dgrad+Wgrad + 随机 Hadamard (H16) | **不收敛** |
| Fprop+Dgrad+Wgrad + **确定性 Hadamard** (H16/H32) | **8–9%（恢复稳定）** |

结论：不稳定的根源是 **Wgrad 路径上结构化的 micro-scaling 误差（outlier 主导）**，不是"随机性不足"；因此加噪声（SR、随机旋转）没用，只有去掉噪声、用固定的确定性旋转把 outlier 能量打散才有效。

---

## 2. 两条链路的设计对照（当前状态）

| 维度 | 论文 (AMD/PSU, MI355X) | Lumen (本仓库, MI350X gfx950) |
|---|---|---|
| 模型/任务 | Llama 3.1-8B, MLPerf C4, 目标 val ppl ≤ 3.3 | Qwen3-0.6B（已验证）/ Qwen3-8B（**已收敛，5000 步稳定**） |
| 硬件 | MI355X 原生 FP4 Tensor Core | MI350X (gfx950) 原生 FP4 ASM (`v_cvt_scalef32_*_pk_fp4_*`) |
| GEMM 后端 | AMD ROCm Transformer Engine | AITER `gemm_afp4wfp4` |
| Fprop 量化 | MXFP4 | 权重 2D(32×32) RTN，激活 1D(1×32) RTN |
| Dgrad 量化 | MXFP4 | dY 用 1×32 **SR**；权重从 BF16 master (`ctx.weight_ref`) 重量化（转置），**无 Hadamard** |
| Wgrad 量化 | MXFP4 + **确定性 Hadamard** | dY^T, X^T 均先做**确定性 Hadamard（全 +1, G=16）**，再 **SR** 量化 1×32 |
| Hadamard 应用范围 | Fprop+Dgrad+Wgrad 均可套（附录C 证明三阶段可对消），最终全链路开启 | **只在 Wgrad**（kernel 外做 BF16 Hadamard，Fprop/Dgrad 外部旋转会断梯度链，见 §4） |
| Hadamard 随机性 | 结论：**必须确定性**（随机符号在 Wgrad 全量化时不收敛） | **确定性（全 +1）** —— `_get_mxfp4_rht_sign()` 返回 `torch.ones(...)`，与论文结论一致 ✅ |
| Wgrad 舍入 | SR 单独用不收敛 → 靠确定性 Hadamard 稳定 | 两操作数用 **SR**，但叠加在**确定性** Hadamard 上（非论文中失败的"随机+SR"组合），实测稳定 |
| Hadamard block 大小 | H16 最优（比 H32 快 8%），H32 也可行 | **g=16**（`_MXFP4_RHT_G = 16`）✅ |
| 末尾层保高精度 | **论文未涉及**：所有 transformer linear layer 全量化，无按层豁免（基线是 FP8） | **默认保留末尾 `round(0.15·L)` 层 BF16**（8B: 5/36 层）—— 依据是 NVFP4 论文 §4 / 附录 E.2 |
| 量化粒度策略 | 附录D：2D 适合权重，1D 行适合激活 | 权重 2D(32×32)，激活 1D(1×32) —— **与论文附录D 一致** ✅ |
| 端到端结果 | 全链路 MXFP4+确定性 Hadamard：比 FP8 快 9–10%，token 开销 8–9% | 0.6B 收敛 (Δloss +0.045)；8B 收敛 (val_loss 7.07→5.74, 5000 步)。**当前 8B 比 BF16 快 6.7%**（869.4 vs 928.0 ms，见 `mxfp4_optimization_report.md`） |

---

## 3. Lumen 如何落地论文配方 + 结果

解决 Lumen 8B 收敛问题的三条关键措施，均已应用（对照代码）。前两条出自本文，第三条出自 NVFP4 论文：

**(1) 确定性 Hadamard（非随机符号）** — WGrad 沿 reduction dim M 做 blockwise Hadamard，两操作数共用同一个 H，在 GEMM 内对消 `(dY^T H)(X^T H)^T = dY^T X`：

```2080:2104:lumen/ops/quantize/linear.py
                    # --- WGrad: dW = fused_HQ(dY^T) @ fused_HQ(X^T)^T ---
                    rht_g = _MXFP4_RHT_G
                    _rht_ok = (M % rht_g == 0)

                    # Left as a view: hadamard_quant_mxfp4 indexes through both
                    # strides, so it reads the transpose directly and the
                    # (N_out, M) copy never happens.
                    grad_t = grad_flat.t()
                    # Fused, so no separate BF16 (M, K) dequant buffer is written.
                    # It also lands dense, which the non-RHT quantizer below needs.
                    input_t = dequant_transpose_mxfp4(
                        input_data, input_scale, block_size=mxfp4_block,
                    )

                    if _rht_ok:
                        sign_m = _get_mxfp4_rht_sign(grad_flat.device)
                        grad_t_fp4, grad_t_scale = hadamard_quant_mxfp4(
                            grad_t, sign_m, block_size=mxfp4_block, g=rht_g, use_sr=True,
                        )
                        # NVFP4 §4.4 / E.3: stochastic rounding belongs on the
                        # gradient only. On activations it buys little and can
                        # diverge, so the activation stays round-to-nearest.
                        input_t_fp4, input_t_scale = hadamard_quant_mxfp4(
                            input_t, sign_m, block_size=mxfp4_block, g=rht_g, use_sr=False,
                        )
```

Hadamard 和量化已经融进一个 kernel（`hadamard_quant_mxfp4`），butterfly 在寄存器里做完直接
写 packed FP4，中间那份 BF16 不落地 —— 方向上比论文外部旋转的写法更接近 ROCm TE。

符号向量为确定性全 +1（论文结论：随机符号在 8B+ 会发散）：

```83:88:lumen/ops/quantize/linear.py
def _get_mxfp4_rht_sign(device: torch.device) -> torch.Tensor:
    """Return deterministic Hadamard sign vector (all +1)."""
    global _MXFP4_RHT_SIGN
    if _MXFP4_RHT_SIGN is None or _MXFP4_RHT_SIGN.device != device:
        _MXFP4_RHT_SIGN = torch.ones(_MXFP4_RHT_G, device=device, dtype=torch.float32)
    return _MXFP4_RHT_SIGN
```

**(2) G=16** — `_MXFP4_RHT_G = 16`（论文：H16 比 H32 快 8% 且同等稳定）。

**(3) 末尾层保 BF16** — MXFP4 预训练脚本默认保留末尾 ~15% 层为 BF16（`pretrain_qwen3_mxfp4.py`）。⚠️ 这条**不是本文的结论**，而是 NVFP4 论文 §4 / 附录 E.2 的（"末尾线性层最敏感，保留 <15% 在 BF16/MXFP8"）。本文对所有 linear layer 全量化，"最敏感"指的是 **Wgrad 这条通路**，不是层的位置。

**结果**：8B MXFP4 从原先的 step ~1275–3600 崩溃（loss spike 到 11.94），改为 **5000 步稳定收敛，val_loss 7.07→5.74，零发散**（lr=1e-4, warmup=50, FSDP2 8×MI350X）。详见 `docs/mxfp4_status_report.md` §4.4 与 `docs/mxfp4_debug_flow.md`。

> 注：Lumen 的 Wgrad 在确定性 Hadamard 之上对**梯度**叠加 SR，对**激活**用 RTN
> （`40e2691` 之前两者都是 SR，改动理由见 NVFP4 论文 §4.4 / 附录 E.3：SR 只属于梯度）。
> 这与论文中"**随机** Hadamard + SR 不收敛"不冲突——失败的是随机符号，而非 SR 本身。

---

## 4. 其它一致 / 不一致点

- **一致**：微缩放 block=32、E8M0 标准定义；权重 2D block + 激活 1D 行块的选型与论文附录D 结论一致；确定性 Hadamard、末尾层保 BF16 均已对齐。
- **不一致（Hadamard 范围）**：论文覆盖 Fprop+Dgrad+Wgrad 三阶段；Lumen **只在 Wgrad**。原因：论文的 ROCm TE 在 **kernel 内部**做旋转（寄存器级 H cancel），Lumen 在 **kernel 外部**做 BF16 Hadamard，Fprop/Dgrad 若外部旋转会导致 backward 的 grad_output 缺配套反旋转、梯度链断裂。当前 Wgrad-only 已足够稳定 8B。
- **不一致（融合/性能）**：论文用 ROCm TE 的 GEMM prologue fusion（H+Quant+GEMM 单 kernel），比 FP8 快 9–10%；Lumen 已把 Hadamard+Quant 融成一个 kernel，但**量化和 GEMM 仍是两次独立 launch**。当前 8B（seq 2048 × mbs 4 × 8 卡）实测 **比 BF16 快 6.7%**（869.4 vs 928.0 ms），代价是显存 15.30 → 20.90 GB（见 `docs/mxfp4_optimization_report.md`）。
- **不一致（框架/规模）**：论文 Llama-3.1-8B under MLPerf；Lumen 自研框架 + AITER，验证到 Qwen3-0.6B/8B 收敛 + 12/12 算子级对比。

---

## 5. 后续动作

1. ~~**性能**：Lumen 侧融合 Hadamard+Quant kernel~~ **已完成**（`hadamard_quant_mxfp4`）。剩下的是向 AITER 提 GEMM prologue fusion 需求（达到论文性能水平的硬依赖）。
2. ~~**可选消融**：确定性 Hadamard + RTN vs + SR~~ 已按 NVFP4 论文改成梯度 SR / 激活 RTN（`40e2691`）；两者都 SR 的旧配置也稳定收敛过 5000 步，所以这项改动是按论文对齐，不是修 bug。
3. **可选扩展**：评估是否需要把旋转保护扩展到 Dgrad（目前 Dgrad 只有 SR、无旋转）。
4. **8B 长程验证**：当前 5000 步稳定，可延长 token horizon 观察后期稳定性（论文强调大模型/长程才暴露问题）。
