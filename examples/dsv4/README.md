# DSV4 4-layer Megatron Pretrain (MI308X)

在 8×MI308X 上跑 DSV4 Flash 4-layer 的 **native Megatron pretrain smoke**：
`torchrun` + Lumen `get_dsv4_spec`，**不依赖 Miles `train.py` / Ray**。

模型实现：`lumen/models/dsv4/`  
入口脚本：`pretrain_dsv4.py`

---

## 前置条件

| 项目 | 说明 |
|------|------|
| 硬件 | 8× MI308X，宿主机可访问 `/dev/kfd` |
| Docker | 已安装，当前用户可运行 |
| 基础镜像 | `lumen/tests:latest`（构建 DSV4 镜像用） |
| Miles 镜像 | `rlsys/miles:rocm7.2-mi308x`（提取 bootstrap 依赖用） |
| Miles 仓库 | 默认 `/home/leiwu/miles`，需挂载进容器（checkpoint 转换 + Megatron router hook） |
| 磁盘 | 模型目录、日志、TVM cache；bootstrap 约数 GB；`.bootstrap-build` staging 约 4GB（可删） |

默认路径（可通过环境变量覆盖）：

```text
LUMEN_DIR=/home/leiwu/Lumen
MILES_DIR=/home/leiwu/miles
MODEL_DIR=/mnt/data/leiwu/models
LOG_DIR=/mnt/data/leiwu/logs
TVM_CACHE_DIR=/mnt/data/leiwu/tvm-cache
BOOTSTRAP_DIR=/mnt/data/leiwu/lumen-dsv4-bootstrap
```

---

## 目录说明

```text
examples/dsv4/
├── README.md                      # 本文档
├── build_dsv4_lumen_image.sh      # 构建 lumen/dsv4-lumen:mi308x
├── prepare_bootstrap.sh           # 从 Miles 镜像提取 Megatron/tile_kernels 等
├── run_dsv4_pretrain.sh           # 宿主机 Docker launcher
├── run_dsv4_pretrain_inner.sh     # 容器内 torchrun 入口
├── pretrain_dsv4.py               # Megatron pretrain 主程序
├── prepare_dsv4_checkpoint.py     # HF → BF16 → torch_dist（可选）
├── dsv4_4layer_megatron_args.sh   # 4-layer 模型与并行参数
├── setup_container_env.sh         # 容器内 Megatron / tile_kernels 环境
├── bootstrap_env.sh               # bootstrap PYTHONPATH 设置
└── .bootstrap-build/              # 镜像构建 staging（gitignore，可删）
```

---

## 完整流程

### 1. 构建镜像（首次或依赖变更时）

```bash
cd /home/leiwu/Lumen
bash examples/dsv4/build_dsv4_lumen_image.sh
```

脚本会自动：

1. 若 `${BOOTSTRAP_DIR}/.ready` 不存在 → 运行 `prepare_bootstrap.sh`  
   从 `rlsys/miles:rocm7.2-mi308x` 提取 Megatron-LM、tile_kernels、tilelang 等到 `${BOOTSTRAP_DIR}`
2. 拷贝 bootstrap → `examples/dsv4/.bootstrap-build/`（Docker build context staging）
3. `docker build -f Dockerfile.dsv4-lumen` → 产出 **`lumen/dsv4-lumen:mi308x`**

`.bootstrap-build/` 只是 build 中间目录，已在 `.gitignore`；删了不影响代码，下次 build 会重建。

也可单独准备 bootstrap（不 build 镜像）：

```bash
bash examples/dsv4/prepare_bootstrap.sh
```

### 2. 准备 checkpoint（首次无 torch_dist 时）

目标路径（容器内）：

```text
/root/models/DeepSeek-V4-Flash-FP8-4layer_torch_dist_hc2
# 或与 Miles smoke 相同：DeepSeek-V4-Flash-FP8-4layer_torch_dist（自动 fallback）
```

默认训练参数与 Miles smoke（`smoke_test_dsv4_mi308x.py`）对齐：

| 参数 | Lumen pretrain | Miles smoke |
|------|----------------|-------------|
| `DSV4_HC_MULT` | 2 | 2（MI308X） |
| `GBS` | 256（32×8） | 256 |
| `MBS` | 1 | 1 |
| `SEQ_LEN` | 2048（mock 固定长度） | 变长 rollout |
| 并行 | TP=8, EP=8, PP=1, CP=1 | 同左 |
| recompute | full / uniform / 1 layer | 同左 |
| optimizer | adam, lr=1e-6, … | 同左（Miles 另开 CPU offload） |

若已有 `hc4` checkpoint，可显式指定：`DSV4_HC_MULT=4 LOAD_CKPT=1 ...`

若宿主机已有该目录（含 `latest_checkpointed_iteration.txt`），可跳过本步。

**方式 A — 跑 pretrain 时自动转换**（需挂载 `MILES_DIR`）：

```bash
cd /home/leiwu/Lumen
bash examples/dsv4/run_dsv4_pretrain.sh
# SKIP_PREPARE 默认为 0，缺 ckpt 时会调 prepare_dsv4_checkpoint.py
```

**方式 B — 手动转换**（容器内或宿主机有 Miles 环境时）：

```bash
DSV4_HC_MULT=2 python examples/dsv4/prepare_dsv4_checkpoint.py
```

流程：HF 权重 → FP8 转 BF16 → Megatron `torch_dist`（用 Miles `convert_hf_to_torch_dist.py`，不用 `train.py`）。

### 3. 运行 pretrain smoke test（推荐命令）

已有 checkpoint 时：

```bash
cd /home/leiwu/Lumen
SKIP_PREPARE=1 LOAD_CKPT=1 TRAIN_ITERS=10 bash examples/dsv4/run_dsv4_pretrain.sh
```

`run_dsv4_pretrain.sh` 会：

- 镜像不存在时自动调用 `build_dsv4_lumen_image.sh`
- 挂载 Lumen 源码、模型、日志、TVM cache、Miles 仓库
- 在容器内执行 `run_dsv4_pretrain_inner.sh` → `torchrun` 8 卡

日志：`/mnt/data/leiwu/logs/lumen_dsv4_pretrain_<timestamp>.log`

### 4. 验证成功

日志末尾应出现：

```text
iteration 10/10
lm loss
=== [done] Lumen DSV4 Megatron pretrain smoke completed ===
```

说明：

- **第一步较慢**（约 4–5 分钟）：TileLang kernel JIT 编译
- GBS=256 时每 iteration 约 256 个 microbatch，耗时与 Miles smoke 同量级（无 Ray 开销会略快）
- 使用 Megatron `--mock-data`，非 Miles rollout 变长数据

---

## 一键流程（已有 checkpoint）

```bash
cd /home/leiwu/Lumen

# 构建镜像（只需一次）
bash examples/dsv4/build_dsv4_lumen_image.sh

# 跑 10 step smoke
SKIP_PREPARE=1 LOAD_CKPT=1 TRAIN_ITERS=10 bash examples/dsv4/run_dsv4_pretrain.sh
```

---

## 常用环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `IMAGE` | `lumen/dsv4-lumen:mi308x` | Docker 镜像 |
| `TRAIN_ITERS` | `10` | 训练步数 |
| `DSV4_HC_MULT` | `2` | 与 Miles smoke 一致；须与 checkpoint 匹配 |
| `GBS` | `256` | `ROLLOUT_BATCH_SIZE × N_SAMPLES_PER_PROMPT` |
| `ROLLOUT_BATCH_SIZE` | `32` | Miles smoke 对齐 |
| `N_SAMPLES_PER_PROMPT` | `8` | Miles smoke 对齐 |
| `MBS` | `1` | micro-batch size |
| `SEQ_LEN` | `2048` | mock 固定 seq |
| `LOAD_CKPT` | `1` | 是否加载 torch_dist |
| `SKIP_PREPARE` | `0` | `1` 跳过 HF→torch_dist |
| `DSV4_ENABLE_RECOMPUTE` | `1` | activation recompute |
| `LUMEN_DSV4_LINEAR_FP8` | `0` | Lumen FP8 linear |
| `V4_SPARSE_MLA_BACKEND` | `tilelang` | sparse MLA 后端 |
| `V4_INDEXER_BLOCK_N` | `64` | MI308X LDS 限制，勿随意改大 |
| `MILES_DIR` | `/home/leiwu/miles` | Miles 仓库路径 |
| `MODEL_DIR` | `/mnt/data/leiwu/models` | 模型与 checkpoint 目录 |

示例：

```bash
# 随机初始化，不加载 ckpt
LOAD_CKPT=0 SKIP_PREPARE=1 TRAIN_ITERS=5 bash examples/dsv4/run_dsv4_pretrain.sh

# 关闭 recompute
DSV4_ENABLE_RECOMPUTE=0 SKIP_PREPARE=1 LOAD_CKPT=1 bash examples/dsv4/run_dsv4_pretrain.sh

# Lumen FP8 linear
LUMEN_DSV4_LINEAR_FP8=1 SKIP_PREPARE=1 LOAD_CKPT=1 bash examples/dsv4/run_dsv4_pretrain.sh
```

---

## 替代：不 bake 镜像，运行时挂载 bootstrap

使用 `lumen/tests:latest`，运行时挂载 `${BOOTSTRAP_DIR}`：

```bash
IMAGE=lumen/tests:latest bash examples/dsv4/run_dsv4_pretrain.sh
```

需先 `prepare_bootstrap.sh`；首次启动会慢一些（容器内拷贝 site-packages）。

---

## 架构简述

```text
run_dsv4_pretrain.sh (host)
  └─ docker run lumen/dsv4-lumen:mi308x
       └─ run_dsv4_pretrain_inner.sh
            ├─ setup_container_env.sh  → Megatron + tile_kernels
            ├─ prepare_dsv4_checkpoint.py  (可选)
            └─ torchrun pretrain_dsv4.py
                 └─ megatron.training.pretrain
                      └─ lumen.models.dsv4.megatron.spec.get_dsv4_spec
```

Megatron 参数：`--spec lumen.models.dsv4.megatron.spec get_dsv4_spec`

并行布局（8 卡）：TP=8, EP=8, PP=1, CP=1；batch 见 `dsv4_4layer_megatron_args.sh`（默认 GBS=256）。
