# 论文对比：《Pretraining Large Language Models with NVFP4》(NVIDIA) vs Lumen MXFP4 链路

- 论文：arXiv:2509.25149v2（NVIDIA，2026-03）
- PDF 已下载：`docs/papers/nvfp4_pretraining_2509.25149.pdf`
- 姊妹篇（AMD/PSU，MXFP4 on MI355X）：arXiv:2605.09825，对比见 `docs/papers/mxfp4_paper_vs_lumen_comparison.md`
- Lumen 侧参照：`lumen/ops/quantize/linear.py`、`lumen/ops/quantize/ops.py`、`lumen/kernels/mxfp4.py`、`lumen/quantize/`、`docs/mxfp4_training_report.md`

> 说明：本仓库实现的是 **MXFP4**（OCP microscaling：FP4 E2M1 + E8M0 block scale，block=32），**没有单独的 NVFP4 格式代码**。本文档用 NVFP4 论文的方法学 / 超参作为参照系，核对 Lumen 现状。

---

## 1. 超参数推荐

### 1.1 NVFP4 论文给出的三种尺寸配置（Appendix A）

| 超参 | 8B（用于对比 MXFP4） | 12B（主验证模型） | 1.2B（消融用） |
|---|---|---|---|
| 架构 | Hybrid Mamba-Transformer，52 blocks（4 Attn / 24 FFN / 24 Mamba-2） | 62 blocks（6 Attn / 28 FFN / 28 Mamba-2） | 标准 Transformer，20 blocks |
| hidden / FFN dim | 4096 / 21504 | 5120 / 20480 | 2048 / 6144 |
| 注意力 | GQA，32 Q / 4 KV heads | 40 Q / 8 KV heads | 16 Q / 8 KV heads（RoPE） |
| 训练 tokens | 1T | 10T | 1T |
| sequence length | 8192 | 8192 | 8192 |
| batch size | 768 | 736 | 768 |
| LR 调度 | WSD，恒定 **8.0e-4** → **8.0e-6**（最后 15%） | WSD，**4.5e-4** → **4.5e-6**（最后 20%） | WSD，**1.2e-3** → **1.2e-5**（最后 15%） |
| Adam (β1, β2) | 0.9, 0.95 | 0.9, 0.95 | 0.9, 0.95 |
| weight decay | 0.1 | 0.1 | — |
| 高精度层（BF16） | 最后 8 个 block（≈15%） | 前 2 + 后 8 个 block（16%） | 消融用，不固定 |
| 参考基线 | BF16 | FP8 | BF16 |

### 1.2 通用配方（Section 4，适用于所有尺寸）

1. **~15% 敏感线性层保留高精度（BF16/MXFP8），且主要放在网络末尾**。
2. **对 Wgrad 的输入做 16×16 Random Hadamard 变换**。
3. **权重用 2D（16×16）block scaling；激活/梯度用 1D（1×16）**。
4. **梯度用随机舍入（SR）；权重/激活用 round-to-nearest-even（RNE）**。

---

## 2. 超参 / 消融实验结论

### 2.1 NVFP4 论文
- **Hadamard 矩阵尺寸**：选 `d=16`。小模型（1.2B）2×2 / 4×4 / 16×16 / 128×128 几乎无差别；12B 上 4×4 明显变差、128×128 只微弱提升。**小模型的结论不能外推到大模型**（作者反复强调）。
- **Hadamard 只加在 Wgrad**：加到 Fprop/Dgrad 反而掉点。
- **随机符号向量**：整个训练共用**一个固定 seed 就够**；随机化只在大模型 / 长 horizon 才有收益。
- **随机舍入 SR 只用于梯度**：用在权重或激活上会发散。
- **层敏感度**：**末尾几层最敏感**；只保留开头几层高精度无效，必须包含末尾层。
- **末期切高精度**：LR 衰减前后（10T 里 8.2T 处）把**前向**切到 BF16，可把相对 loss 差从 ~1.5% 收到 ~0.5%，仅占约 6% 计算量。loss gap 主要来自前向量化。
- **NVFP4 vs MXFP4**：8B 上 MXFP4 需**多训 36% tokens（1.36T vs 1T）**才能追上 NVFP4；MXFP4 因块大小 32，Hadamard 用 `d=32`。

### 2.2 ⚠️ 两篇论文的关键矛盾
在**如何稳定 Wgrad** 上，两篇结论相反：

| | NVFP4 (NVIDIA) | MXFP4 (AMD, arXiv:2605.09825) |
|---|---|---|
| 随机舍入 SR | **必需**（梯度上用它才收敛） | **无效**（全流程不收敛） |
| Hadamard | **随机** Hadamard（大模型才需随机化） | **确定性** Hadamard（随机符号反而有害） |

AMD 那篇自陈：FP4 recipe **非通用**，随模型/数据集/格式而变。可能原因：格式不同（NVFP4 块 16 + E4M3 scale vs MXFP4 块 32 + E8M0）、模型不同（Nemotron hybrid vs Llama 3.1）、规模不同（10T vs MLPerf 短程）。**Lumen 做 MXFP4 + MI 系列硬件，与 AMD 那篇场景更贴近，优先参考它的"确定性 Hadamard"结论。**

---

## 3. 逐条核对：论文方法学 vs Lumen 现状

> 代码位置以 `lumen/ops/quantize/linear.py`、`ops.py`、`kernels/mxfp4.py` 为准（本次已实读确认）。

| 方法学要素 | 论文推荐 | Lumen 现状 | 状态 |
|---|---|---|---|
| **FP4 格式** | NVFP4（块 16，E4M3 scale，两级+FP32 per-tensor） | MXFP4（块 32，E8M0 scale，**单级**，无 per-tensor FP32） | ⚠️ 格式不同（Lumen=MXFP4） |
| **Block 大小** | 16（NVFP4） | **32**（微缩放 block `mxfp4_block=32`；注：Hadamard 的 G 是独立参数，现为 16） | ⚠️ 与 MXFP4 定义一致，非 NVFP4 |
| **两级/FP32 per-tensor scale** | 有 | **无**（仅 E8M0 block scale） | ❌ 缺失（MXFP4 无此机制） |
| **权重 2D block scaling** | 16×16 | **32×32**（`convert_to_mxfp4_2d`） | ✅ 有（尺寸随 MXFP4=32） |
| **激活/梯度 1D scaling** | 1×16 | **1×32**（`convert_to_mxfp4(axis=-1)`） | ✅ 有 |
| **Hadamard 应用范围** | 仅 Wgrad | **仅 Wgrad**（Fprop/Dgrad 明确无 RHT，见 `linear.py:2065-2078`（DGrad）与 `2080-2113`（WGrad）） | ✅ 一致 |
| **Hadamard 随机性** | NVFP4 建议随机（大模型）；AMD 建议确定性 | **确定性（全 +1）**——`_get_mxfp4_rht_sign` 返回 `torch.ones`，注释引用 arXiv:2605.09825（`linear.py:74-88`） | ✅ 采用 AMD 结论（已修复旧的随机 bug） |
| **Hadamard 矩阵尺寸** | 16（NVFP4）/ 论文发现 H16 与 H128 相当 | **g=16**（`_MXFP4_RHT_G=16`，`linear.py:80`；AMD 发现 H16 比 H32 快 8%） | ✅ 已采用 16 |
| **SR 选择性（仅梯度）** | 仅梯度 SR，权重/激活 RNE | **完全一致**：Fprop 权重/激活 RTN；Dgrad 的 dY 用 SR、权重复用 forward 的 RTN 结果；Wgrad 的 dY^T 用 SR、**激活 X^T 用 RTN**（`linear.py:2094-2113`） | ✅ 一致（`40e2691` 之前 Wgrad 两个操作数都是 SR，与本条不符） |
| **Hadamard+量化融合** | ROCm TE 在 kernel 内做（NVFP4 论文用 TE） | **已融合**：`hadamard_quant_mxfp4` 单 kernel 完成 butterfly + 量化，中间 BF16 不落地 | ✅ 方向一致（GEMM prologue 仍未融合） |
| **末尾敏感层保高精度** | ~15%，末尾为主 | MXFP4 预训练脚本**默认开启**：末尾 `round(0.15·L)` 层 BF16（`pretrain_qwen3_mxfp4.py`）；底层 `first_last_layers_bf16`（`config.py` 全局默认 False） | ✅ 已启用（末尾优先） |
| **末期切高精度（BF16 收尾）** | 推荐（收窄 loss gap） | **无**该机制 | ❌ 缺失 |
| **训练链路（Fprop/Dgrad/Wgrad GEMM）** | Transformer Engine | `QuantizedLinearFunction` + AITER `gemm_afp4wfp4`，完整前反向 | ✅ 有 |
| **NVFP4 格式实现** | — | **无**（仅注释引用） | ❌ 未实现 |

图例：✅ 已覆盖　⚠️ 部分/有差异　❌ 缺失

---

## 4. 差距与建议动作

按"对当前 8B 收敛问题的影响 / 落地成本"排序：

1. **末尾敏感层保高精度（已落地）** — 论文（两篇都强调）末尾层对 FP4 最敏感。MXFP4 预训练脚本已默认保留末尾 ~15% blocks 为 BF16；8B 借此 + H16 已稳定收敛 5000 步（见 `docs/mxfp4_status_report.md` / `mxfp4_training_report.md`）。✅ 这是论文里对稳定性贡献最大的单一因素，已验证有效。
2. **Hadamard 尺寸已用 16（已落地）** — Lumen `_MXFP4_RHT_G=16`（原 32）。AMD 论文 H16 比 H32 快 8% 且同样稳定，NVFP4 论文 16 与 128 相当。✅
3. **末期切 BF16 收尾** — 若 8B 稳定后仍与 BF16/FP8 有 loss gap，可在 LR 衰减前把 Fprop 切回 BF16（论文只需约 6% 计算量即可收窄 gap）。目前 Lumen 无此机制，需新增。
4. **SR 只用于梯度（已落地）** — `40e2691` 把 Wgrad 的激活操作数从 SR 改成 RTN，与论文 §4.4 / 附录 E.3 对齐。注意这是**按论文对齐**，不是修 bug：两个操作数都用 SR 的旧配置也稳定收敛过 5000 步。✅
5. **末期切 BF16 收尾** — 若 8B 稳定后仍与 BF16 有 loss gap，可在 LR 衰减前把 Fprop 切回 BF16（论文只需约 6% 计算量即可收窄 gap）。目前 Lumen 无此机制，需新增。
6. **（可选）两级 FP32 per-tensor scale** — NVFP4 的核心增益之一。若要逼近 NVFP4 精度而非纯 MXFP4，需要引入 per-tensor FP32 scale；工程量较大，非当前收敛问题的必需项。
7. **MXFP4 linear 的前反向数值测试（已落地）** — `tests/test_mxfp4_backward_optimization.py` 覆盖 2D block scale 转置不变性（逐位）、fused Hadamard+Quant 等价性（实测 SNR 29.4 dB）、权重缓存正确性（逐位）、前反向对 BF16 参考（实测前向 15.3 dB / WGrad 14.2 dB，断言下限 5 / 3 dB）和 50 步收敛；`035431e` 又加了 `tests/ops/test_quantize.py::test_mxfp4_backward_gradients_track_the_bf16_reference`（dW 12.9 dB、dX 13.7 dB，断言下限 11 dB）。✅

---

## 5. 备注

- 本仓库另一份 `docs/papers/mxfp4_paper_vs_lumen_comparison.md` 写于**随机 Hadamard 修复之前**，其中"§3 随机符号是 bug、需改确定性"的诊断**已落地**（`_get_mxfp4_rht_sign` 现返回全 +1），该文已按此更新。
- Lumen 端到端验证进度：0.6B 收敛正常（Δ val_loss +0.045）；8B MXFP4 用 H16 + 末尾 5/36 层 BF16
  跑完 5000 步稳定收敛（val_loss 5.74）。性能上 8B 当前**比 BF16 快 6.7%**
  （869.4 vs 928.0 ms，seq 2048 × mbs 4 × 8 卡），代价是显存 15.30 → 20.90 GB；
  逐项实测见 `docs/mxfp4_optimization_report.md`。
