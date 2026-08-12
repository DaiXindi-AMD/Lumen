# Debug Environment Quick Deploy Guide

Target: AMD GPU node (gfx94x/gfx950) with ROCm pre-installed.

Script source: `scripts/setup_env.sh` (in this repo)

---

## On a brand new machine, run two commands:

```bash
# 1. Install Claude Code (interactive, needs confirmation)
curl -fsSL https://claude.ai/install.sh | bash -s stable

# 2. One-click everything else (zsh, plugins, repos, Claude settings, triton shim)
bash <(curl -fsSL https://raw.githubusercontent.com/DaiXindi-AMD/Lumen/main/scripts/setup_env.sh)

# 3. Enter new shell
exec zsh -l
```

To use a different API key:
```bash
LUMEN_API_KEY="your-key-here" bash <(curl -fsSL https://raw.githubusercontent.com/DaiXindi-AMD/Lumen/main/scripts/setup_env.sh)
```

If you already have Lumen cloned:
```bash
bash ~/Lumen/scripts/setup_env.sh
```

Idempotent — re-run anytime (reboot clears `/tmp/tritonshim`, re-run restores it).

---

## Quick Reference

### Repository Layout

| Repo | Path | Remote | Purpose |
|------|------|--------|---------|
| **Lumen** | `~/Lumen` | `origin`: ZhangDanyang-AMD/Lumen, `mine`: DaiXindi-AMD/Lumen | Main project |
| **AITER** | `~/aiter` | `origin`: ROCm/aiter | AMD GPU kernels (Triton, CK, ASM) |
| **AITER** (submodule) | `~/Lumen/third_party/aiter` | ZhangDanyang-AMD/aiter (`lumen/triton_kernels` branch) | Pinned kernel version for Lumen |
| **TorchAO** | `~/ao` | `origin`: pytorch/ao | Quantization library |

### Verified Environment

| Component | Version |
|-----------|---------|
| ROCm | 7.8.0 |
| PyTorch | 2.13.0+rocm7.2 |
| TorchAO | 0.18.0+ |
| Python | 3.10.12 |
| Claude Code | 2.1.202 (stable) |

### Post-Setup Checklist

- [ ] Run `source ~/.zshrc` or open a new terminal
- [ ] Run `claude` and accept the custom API key prompt (type "dummy")
- [ ] Verify GPU: `rocm-smi` shows MI300X/MI350 cards
- [ ] Verify torch: `python3 -c "import torch; print(torch.cuda.is_available())"`
- [ ] Verify AITER: `python3 -c "import aiter; print('ok')"`
- [ ] Verify TorchAO: `python3 -c "import torchao; print(torchao.__version__)"`

### Triton Shim 说明

**问题**：Triton 3.7 的 `JITCallable.__init__` 用 `inspect.getsourcelines()` 解析源码时，AITER 的 `@triton.heuristics` 装饰器会导致返回的源码片段里缺少 `def` 行，触发 `AttributeError` 崩溃。

**原理**：`PYTHONPATH=/tmp/tritonshim:...` 利用 Python 的 `sitecustomize.py` 机制——Python 解释器启动时会自动执行 `sys.path` 中找到的第一个 `sitecustomize.py`。这个 shim 在 Triton 加载前 monkey-patch 了两个函数：
- `get_def_col_number` → 异常时返回 1（仅影响调试时的源码行号定位）
- `JITCallable.__init__` → `AttributeError` 时走手动初始化

**新机上需要吗**：只要 **Triton 3.7 + AITER fork** 组合就需要。setup 脚本已自动创建。`/tmp` 重启会清空，重跑 `setup_env.sh` 即可恢复。

**运行 MXFP4 测试**：
```bash
cd ~/Lumen
PYTHONPATH=/tmp/tritonshim:$PWD/third_party/aiter \
  python -m pytest tests/ops/test_quantize.py -v -k mxfp4 -p no:cacheprovider
```

`PYTHONPATH` 中两个路径的作用：
- `/tmp/tritonshim` — 加载 sitecustomize.py 修复 Triton bug
- `$PWD/third_party/aiter` — 让 Python 能直接 `import aiter`（子模块版本，非 pip 安装的 `~/aiter`）

### Updating the API Key

If the key changes, update in **two** places:

```bash
# 1. Shell env
sed -i "s/Ocp-Apim-Subscription-Key: .*/Ocp-Apim-Subscription-Key: NEW_KEY_HERE/" ~/.zshrc

# 2. Claude settings
sed -i "s/Ocp-Apim-Subscription-Key: .*/Ocp-Apim-Subscription-Key: NEW_KEY_HERE\"/" ~/.claude/settings.json

source ~/.zshrc
```
