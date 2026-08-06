# DSV4 Flash 全模型 Runbook（p14 + p38，MI308X）

操作手册：在 **p14（head）+ p38（worker）** 双节点 16 GPU 上跑 Lumen native GRPO full finetune（43 层）。详细脚本说明见 [README.md](./README.md)。

---

## 1. 集群与环境

| 节点 | Hostname | IP | 角色 |
|------|----------|-----|------|
| p14 | `banff-ccs-aus-p20-14` | `10.194.132.29` | head（`NODE_RANK=0`） |
| p38 | `banff-ccs-aus-p20-38` | `10.194.132.28` | worker（`NODE_RANK=1`） |

| 项目 | 值 |
|------|-----|
| Docker 镜像 | `lumen/dsv4-lumen:mi308x` |
| SSH key | `~/.ssh/id_ed25519_conductor` |
| 并行 | TP=4, PP=4, EP=4（11+11+11+10 层） |
| 网络 | `NCCL_SOCKET_IFNAME=ens14np0`, `GLOO_SOCKET_IFNAME=ens14np0` |

**p38 上还需**：`~/Lumen`、`~/miles`、`~/TileKernels`（与 p14 同步）。

---

## 2. 路径（NFS）

默认由 `dsv4_paths.sh` 解析（`DATA_ROOT=/nfs/data/leiwu`）：

| 用途 | 宿主机路径 | 容器内路径 |
|------|-----------|-----------|
| 模型目录 | `/nfs/data/leiwu/models` | `/root/models` |
| 日志目录 | `/nfs/data/leiwu/logs` | — |
| Lumen 代码 | `~/Lumen` | `/workspace/Lumen` |

### 2.1 预训练 checkpoint（finetune **加载**）

| 路径 | 说明 |
|------|------|
| `/nfs/data/leiwu/models/DeepSeek-V4-Flash-FP8_torch_dist` | 主 checkpoint（torch_dist） |
| `/nfs/data/leiwu/models/DeepSeek-V4-Flash-FP8_torch_dist_hc4` | fallback（主路径不存在时） |

容器内等价：`/root/models/DeepSeek-V4-Flash-FP8_torch_dist`  
校验：`latest_checkpointed_iteration.txt` 存在即可。

### 2.2 GRPO rollout 数据

| 路径 | 说明 |
|------|------|
| `/nfs/data/leiwu/models/fake_rollout.pt` | debug-train-only 假 rollout（256 样本，GBS=256） |

rank0 首次运行可自动生成；已存在则跳过。

### 2.3 Finetune **输出** checkpoint

**默认不保存**。脚本未传 `--save`，`--save-interval 1000000` 等价于关闭。  
本次 smoke 仅 `--load` 预训练权重并做 GRPO 更新，不会写回 NFS。

如需保存 finetune 权重，在 `dsv4_finetune_common.sh` 的 `DSV4_FINETUNE_TORCHRUN_ARGS` 中增加例如：

```bash
--save /root/models/DeepSeek-V4-Flash-FP8-finetune_torch_dist
--save-interval 1
```

（宿主机可见路径：`/nfs/data/leiwu/models/DeepSeek-V4-Flash-FP8-finetune_torch_dist`）

---

## 3. 启动前检查

### 3.1 同步代码到 p38

在 **p14** 执行：

```bash
rsync -az --delete \
  --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
  --exclude 'third_party/aiter/**/build' --exclude '.nfs*' \
  ~/Lumen/ leiwu@10.194.132.28:~/Lumen/ \
  -e "ssh -o BatchMode=yes -i ~/.ssh/id_ed25519_conductor -o IdentitiesOnly=yes"
```

### 3.2 确认 p38 8 卡空闲

```bash
ssh -o BatchMode=yes -i ~/.ssh/id_ed25519_conductor -o IdentitiesOnly=yes \
  leiwu@10.194.132.28 \
  'for i in 0 1 2 3 4 5 6 7; do echo -n "GPU$i: "; \
   rocm-smi -d $i --showmeminfo vram 2>/dev/null | grep Used; done'
```

每张卡 Used 应约 **~300MB** 量级。若有大占用（如 VLLM ~70GB+），需先释放再跑。

### 3.3 清理旧容器

```bash
docker rm -f lumen-dsv4-flash-finetune-node0 lumen-dsv4-flash-finetune-node1 2>/dev/null || true
ssh -o BatchMode=yes -i ~/.ssh/id_ed25519_conductor -o IdentitiesOnly=yes leiwu@10.194.132.28 \
  'docker rm -f lumen-dsv4-flash-finetune-node0 lumen-dsv4-flash-finetune-node1 2>/dev/null || true'
```

---

## 4. 启动：DSV4 43 层 GRPO full finetune

在 **p14** 执行（推荐一键双节点）：

```bash
cd ~/Lumen

MASTER_ADDR=10.194.132.29 WORKER_SSH=leiwu@10.194.132.28 \
SKIP_PREPARE=1 LOAD_CKPT=1 GBS=256 NUM_ROLLOUT=10 DSV4_HC_MULT=4 \
V4_SPARSE_MLA_BACKEND=tilelang MHC_BACKEND=triton \
OPTIMIZER_OFFLOAD_FRACTION=0.75 \
HSA_OVERRIDE_GFX_VERSION=9.4.2 NCCL_SOCKET_IFNAME=ens14np0 GLOO_SOCKET_IFNAME=ens14np0 \
IMAGE=lumen/dsv4-lumen:mi308x \
bash examples/dsv4/launch_dsv4_flash_finetune_2node.sh
```

### 4.1 关键参数

| 变量 | 默认/推荐 | 说明 |
|------|----------|------|
| `GBS` | `256` | 必须等于 rollout 样本数 |
| `NUM_ROLLOUT` | `10` | GRPO rollout / train iters |
| `DSV4_HC_MULT` | `4` | MHC 乘数 |
| `SKIP_PREPARE=1` | — | 跳过 HF→torch_dist 转换（ckpt 已在 NFS） |
| `LOAD_CKPT=1` | — | preflight manifest 标记（finetune 必 load） |
| `V4_SPARSE_MLA_BACKEND` | `tilelang` | 与 pretrain 成功配置一致 |
| `MHC_BACKEND` | `triton` | 需挂载 TileKernels |
| `OPTIMIZER_OFFLOAD_FRACTION` | `0.75` | CPU Adam offload |

### 4.2 单节点手动启动（备用）

Head（p14）：

```bash
cd ~/Lumen
NODE_RANK=0 MASTER_ADDR=10.194.132.29 \
SKIP_PREPARE=1 LOAD_CKPT=1 GBS=256 NUM_ROLLOUT=10 DSV4_HC_MULT=4 \
V4_SPARSE_MLA_BACKEND=tilelang MHC_BACKEND=triton \
OPTIMIZER_OFFLOAD_FRACTION=0.75 \
HSA_OVERRIDE_GFX_VERSION=9.4.2 NCCL_SOCKET_IFNAME=ens14np0 GLOO_SOCKET_IFNAME=ens14np0 \
IMAGE=lumen/dsv4-lumen:mi308x \
bash examples/dsv4/run_dsv4_flash_finetune.sh
```

Worker（p38）：

```bash
cd ~/Lumen
NODE_RANK=1 MASTER_ADDR=10.194.132.29 \
# 其余 env 与 head 相同
bash examples/dsv4/run_dsv4_flash_finetune.sh
```

---

## 5. 参考：全模型 pretrain smoke（mock LM loss）

与 finetune 不同路径，用于 kernel / 多节点连通性验证：

```bash
cd ~/Lumen
MASTER_ADDR=10.194.132.29 WORKER_SSH=leiwu@10.194.132.28 \
SKIP_PREPARE=1 LOAD_CKPT=1 GBS=8 TRAIN_ITERS=10 DSV4_HC_MULT=4 \
V4_SPARSE_MLA_BACKEND=tilelang MHC_BACKEND=triton \
OPTIMIZER_OFFLOAD_FRACTION=0.75 \
HSA_OVERRIDE_GFX_VERSION=9.4.2 NCCL_SOCKET_IFNAME=ens14np0 GLOO_SOCKET_IFNAME=ens14np0 \
IMAGE=lumen/dsv4-lumen:mi308x \
bash examples/dsv4/launch_dsv4_flash_pretrain_2node.sh
```

---

## 6. 日志与监控

| 日志 | 路径 |
|------|------|
| Head 训练 | `/nfs/data/leiwu/logs/lumen_dsv4_flash_finetune_node0_*.log` |
| Worker 训练 | `/nfs/data/leiwu/logs/lumen_dsv4_flash_finetune_node1_*.log` |
| Launch head | `/nfs/data/leiwu/logs/lumen_dsv4_flash_finetune_launch_head_*.log` |
| Launch worker | `/nfs/data/leiwu/logs/lumen_dsv4_flash_finetune_launch_worker_*.log` |
| Preflight | `/nfs/data/leiwu/logs/.dsv4_preflight/runs/<PREFLIGHT_ID>/` |

```bash
# 实时跟踪 head
tail -f /nfs/data/leiwu/logs/lumen_dsv4_flash_finetune_node0_*.log

# 容器状态
docker ps --filter name=lumen-dsv4-flash-finetune
ssh ... leiwu@10.194.132.28 'docker ps --filter name=lumen-dsv4-flash-finetune'
```

### 6.1 训练成功标志

日志中出现 Miles 格式指标，例如：

- `rollout/num_samples`, `rollout/advantages`
- `train/loss`, `train/pg_loss`, `train/grad_norm`
- `perf/actor_train_time`

结束时：

```text
=== [done] Lumen DSV4 Flash full-model native GRPO finetune completed ===
```

### 6.2 正常但耗时的阶段

- **Checkpoint 加载**：大量 `q_norm/kv_norm ... will skip`（ckpt 无这些权重，正常）
- **Optimizer CPU offload 初始化**：GPU 利用率接近 0，host 内存高，日志可能 **数十分钟无新行**（`DISTRIBUTED_TIMEOUT_MINUTES=180`）

---

## 7. 故障排查

| 现象 | 原因 / 处理 |
|------|------------|
| `LOAD_CKPT: unbound variable` / `TRAIN_ITERS: unbound variable` | 使用最新 `preflight_dsv4_flash_multinode.sh`；launch 时设 `LOAD_CKPT=1` |
| `ModuleNotFoundError: sglang.srt`（rollout 生成） | `prepare_dsv4_fake_rollout.py` 会 fallback；或直接使用已有 `fake_rollout.pt` |
| p38 rank OOM（模型 init） | p38 GPU 被 VLLM 等占用；`rocm-smi --showpids` 确认并释放 |
| NCCL hang / 配置不一致 | 两节点必须通过同一 `launch_*_2node.sh` 启动；检查 preflight manifest |
| `NET/IB : Unable to open device mlx5_*` | 容器内 IB 不可用时的 WARN，pretrain/finetune 曾可 fallback 到 socket |
| Worker  unreachable | Conductor SSH：p38 需在 reservation / 安全组内 |

---

## 8. 相关脚本

```text
examples/dsv4/
├── runbook.md                              # 本文档
├── launch_dsv4_flash_finetune_2node.sh     # 双节点 finetune 一键启动（推荐）
├── run_dsv4_flash_finetune.sh              # 单节点 launcher
├── run_dsv4_flash_finetune_inner.sh        # 容器内 torchrun GRPO
├── finetune_dsv4_megatron.py               # Python 入口
├── dsv4_finetune_common.sh                 # batch / ckpt / rollout helpers
├── prepare_dsv4_fake_rollout.py          # fake_rollout.pt
├── preflight_dsv4_flash_multinode.sh       # 双节点配置校验
└── dsv4_paths.sh                           # MODEL_DIR / LOG_DIR 等
```
