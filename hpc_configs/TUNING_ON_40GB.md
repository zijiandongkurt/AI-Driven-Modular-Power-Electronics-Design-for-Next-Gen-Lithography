# GRPO Testing & Tuning on a Single 40GB GPU

A concrete playbook for the case where SLURM only gave you one A100 40GB.
Everything in this doc has been sized against the real memory budget
(see §1) and the actual `RLUpdater` code path.

---

> **Note on defaults**: as of the latest commit, `new_rl_updater.py`
> ships **H100-tuned defaults** (`lora_r=16, lora_alpha=32, lr=2e-5,
> max_length=2048`). The 40GB SLURM script scales these down via env
> vars (`GRPO_LORA_R=8, GRPO_LORA_ALPHA=16, GRPO_LR=1e-5,
> GRPO_MAX_LENGTH=1024`) to fit 4-bit + 40GB. This document still
> uses the **40GB scaled-down values** as the baseline.

## 0. One-time setup

```bash
# 1. clone + env (登录节点上, 一次性, 不抢 GPU)
cd ~
git clone <repo-url> AI-Driven-Modular-Power-Electronics-Design-for-Next-Gen-Lithography
cd AI-Driven-Modular-Power-Electronics-Design-for-Next-Gen-Lithography

# 2. ⚠️  pip install 也别在登录节点跑;先抢一个 CPU job
srun -p rome -A scur2545 --time=00:30:00 --pty bash
bash env.sh
exit                                # 回到登录节点

# 3. 把 HPC override 装上 (不破坏原文件, git checkout 可回滚)
cp hpc_configs/overrides/new_rl_updater.py pipeline/reinforcement_algorithm/new_rl_updater.py
cp hpc_configs/overrides/grpo_trainer.py   pipeline/reinforcement_algorithm/grpo_trainer.py
cp hpc_configs/overrides/rl_demo.py        pipeline/reinforcement_algorithm/rl_demo.py
```

---

## 1. 显存预算 (40GB 卡的真实账)

下面所有 GRPO 步数都基于：

> **Qwen3-14B  +  LoRA(r=8, 7 个 target)  +  bf16 训练精度  +  4 样本/step**

| 项目                     | 不量化 (bf16) | **4-bit NF4 量化** |
|--------------------------|---------------|----------------------|
| 模型权重                 | 28.0 GB       | **7.0 GB**           |
| LoRA 参数 (fp32)         | ~0.1          | ~0.1                 |
| AdamW state (LoRA 只)    | ~0.2          | ~0.2                 |
| 4 样本 × seq=512 激活    | ~12           | ~12                  |
| 4 样本 × seq=1024 激活   | ~24           | ~24                  |
| 4 样本 × seq=2048 激活   | ~48           | ~48                  |
| KV cache / misc          | ~3            | ~3                   |
| **40GB 是否够**          | **全部 OOM**  | seq≤1024 ✅, seq=2048 ❌ |

**结论**：单 40GB 上跑 GRPO 必须开 4-bit 量化。开了之后 seq=1024 留 6GB margin，够稳。

---

## 2. 起步:5 分钟 smoke test

目标:确认环境通了 + 算出真实显存占用 + 拿到第一份 metrics.json。

```bash
# 编辑 SLURM 把 --time 临时降到 0:15:00 (省 budget)
sed -i 's/--time=01:00:00/--time=00:15:00/' hpc_configs/slurm/train_single_40gb.slurm
sbatch hpc_configs/slurm/train_single_40gb.slurm
```

日志里要按顺序看到:

```
✅ Confirmed 40GB-class GPU (40960 MiB) — 4-bit quantization REQUIRED.
[rl_demo] QUANTIZATION=4bit
Loading weights: 100%|...|  339/339 [00:14<00:00]
trainable params: 25,165,824 || all params: 14,025,XXX,XXX || trainable%: 0.1795
Loaded top1: reward=0.XXXX (grpo_reward)     ← 4 行,batch_1 的 4 个候选
...
RL samples: 4
policy_loss: -XX.XXXX
Saved RL LoRA adapter to: ./checkpoints/grpo-lora/final
Saved metrics to pipeline/data/batch_1/grpo_metrics.json
── nvidia-smi after run ──
memory.used, memory.total
33XXX MiB, 40960 MiB                          ← 实际峰值 ≈ 33GB → 还有 7GB 余量
```

**关键看两点**：
1. `trainable% ≈ 0.18` —— LoRA 真的装上了
2. `memory.used` —— 这才是后续调参的"基线"

---

## 3. 调参的两条 ladder

实际上你要调 **两组** 参数:
- **资源 ladder** —— 显存不够时往下退,有富余时往上爬
- **学习 ladder** —— 让 GRPO 真的能学到东西 (reward 上升)

### 3.1 资源 ladder (OOM 修复优先级)

| Step | 改什么 | 怎么改 | 何时用 |
|------|--------|--------|--------|
| 0    | 起手默认 (`seq=1024, n=4, 4bit`) | 直接跑 SLURM | 第一次 |
| 1 ⬇️ | seq 1024 → 768 | SLURM 加 `export GRPO_MAX_LENGTH=768` | smoke 跑出 `CUDA OOM` |
| 2 ⬇️ | seq 768 → 512 | `export GRPO_MAX_LENGTH=512` | step 1 还 OOM |
| 3 ⬇️ | n 4 → 2 | `export GRPO_N_SAMPLES=2` | step 2 还 OOM |
| 4 ⬇️ | LoRA r 8 → 4 | 编辑 `pipeline/reinforcement_algorithm/grpo_trainer.py` 行 47 `lora_r=8` 改 4 | step 3 还 OOM (理论不会) |
| 0 ⬆️ | 反过来,有富余时 | SLURM `nvidia-smi` 看 used < 32GB → 试 seq 2048? **40GB 不建议**,留给学习 ladder | 显存 < 80% 使用 |

**先把这条 ladder 走稳了再动学习参数**——不然每次跑都不知道是 OOM 还是 reward 没动。

### 3.2 学习 ladder (让 reward 真的上升)

#### 阶段 A — 确认能学 (5 步)

跑 5 个 step 看 `policy_loss` 和 `mean_reward` 的趋势。每个 step 是一次 RL update,大概 30–60s。

修改 `pipeline/reinforcement_algorithm/rl_demo.py` 让它跑多步 (我已经在 override 里加好了 `GRPO_FULL_TRAIN` 开关,但你也可以手动 for-loop):

```python
# 把
grpo.train_from_existing_batch(batch_id="batch_1")
# 改成
for step in range(5):
    print(f"\n========== STEP {step+1}/5 ==========")
    grpo.train_from_existing_batch(batch_id="batch_1")
```

期望:

| step | mean_reward                                                                 |
|------|------------------------------------------------------------------------------|
| 1    | 与 batch_1 原始 reward 相同 (LoRA 还没有更新过模型)                            |
| 2-5  | **完全不变 ↗️↘️ 都不应该出现**——因为 prompt + completion 是固定的 batch_1 数据 |

`train_from_existing_batch` 用的是磁盘上**已经生成好的 completions**,LoRA 改了模型但 prompt/completion 是固定的,所以 mean_reward **不会变**。
你应该看的是:
- ✅ `policy_loss` 在 step 之间 **变化** (说明确实在更新参数)
- ✅ `policy_loss` 量级 **没有爆炸** (有限值、不是 NaN/Inf)
- ❌ 如果 `policy_loss` 越来越大或者 NaN: lr 太大 → 降到 5e-6

#### 阶段 B — 关键超参 sweep (用 batch_2 做 holdout)

要看 GRPO 是否真的让生成质量提升,**必须跑 full loop** (generate→validate→reward→update),而 full loop 在 40GB 上需要 ngspice 装好。如果 ngspice 还没通,你能做的"伪 sweep"是:

1. 用 batch_1 跑 N 步 RL,得到 LoRA checkpoint
2. 切回 `train_from_existing_batch`,**但用 batch_2 做评估** (`GRPO_BATCH_ID=batch_2`)
3. 看 batch_2 的 mean_reward 跟原始基准比

按下面的顺序 sweep,**一次只动一个**:

| 优先级 | 参数 | 起点 | 试这些值 | 看什么 |
|--------|------|------|----------|--------|
| 1      | `learning_rate` | 1e-5 | 5e-6, 1e-5, 2e-5, 5e-5 | policy_loss 单调下降但不爆炸 |
| 2      | `lora_r` | 8 | 4, 8, 16 | r=8 vs 16 在 holdout reward 上差多少 |
| 3      | `lora_alpha` | 16 (=2r) | 8, 16, 32 | 整体"撬动力"——通常 alpha = 2r 是稳的 |
| 4      | `kl_beta` | 0.0 | 0.0, 0.01, 0.05 | 当模型开始输出怪东西时开 0.01 抑制 |
| 5      | `max_completion_length` | 512 | 256, 512, 768 | 太短 → 截断 .end 之前 → log_prob 偏差 |

**这些参数都在 `pipeline/reinforcement_algorithm/grpo_trainer.py` 的 `__init__` 里调** (LoRA r/alpha),或者 `new_rl_updater.py` 的 `RLConfig` (lr/kl_beta/max_*)。

#### 阶段 C — 长程训练 (50+ step)

确认 sweep 出最好的 (lr, r, α) 之后,跑长一点的训练:

```bash
# SLURM 里
export GRPO_FULL_TRAIN=1   # 跑完整 generate→validate→...→update 循环
#SBATCH --time=02:00:00    # 2 小时,留够 50 步
```

每 5 步 (`save_every=5`,默认值) 会存一个 LoRA checkpoint 到
`./checkpoints/grpo-lora/step-N/`。把每个 checkpoint 都拿来在 batch_2 上算一次 mean_reward,**画曲线**——这才能真正判断 GRPO 是否在 learn。

⚠️ Full train 需要 ngspice 跑通,40GB 上跑 ngspice 没问题,但你需要先把
`module load ngspice` (或 `pip install ngspice` 或 `conda install -c conda-forge ngspice`,看 Snellius 上哪个可用)  搞定,**而且 PySpice 1.5 跟 ngspice 42 有兼容 bug** (见 `experiment_logs/ngspice_subprocess_test.py` 的发现)。

---

## 4. 监控:你跑训练时该看什么

### 4.1 实时 (job 跑的时候)

另一个 ssh 窗口:

```bash
# 找 jobid
squeue -u $USER

# 进入计算节点 (jobid 替换实际值)
srun --jobid=<JOBID> --pty nvidia-smi -l 2

# 或者 tail 日志
tail -F logs/grpo_single40_<JOBID>.out
```

### 4.2 跑完之后

```bash
# 看每一步的 metrics
cat pipeline/data/batch_1/grpo_metrics.json

# 关键字段
#   policy_loss   ← 应该 step 间变化但有限
#   advantages    ← 4 个值,有正有负 (说明 GRPO 在做 group-relative)
#   mean_reward   ← 用 batch_1 时不变,用 holdout batch 时应该爬升
```

### 4.3 显存

`logs/grpo_single40_*.out` 末尾会有一行 `nvidia-smi after run`,记录峰值。
对比目标:

| 峰值 used   | 含义                                            |
|-------------|--------------------------------------------------|
| < 32 GB     | 很安全,可以试着把 seq 涨到 1280 或 n 涨到 6     |
| 32-37 GB    | sweet spot,保持现状                             |
| 37-40 GB    | 危险,下一次随机一个长 prompt 可能就 OOM,降 seq |
| 显示 OOM    | 走资源 ladder 往下退一格                         |

---

## 5. 失败排查表

| 看到的错误                                                                 | 原因                                          | 修复                                                       |
|----------------------------------------------------------------------------|-----------------------------------------------|------------------------------------------------------------|
| `CUDA out of memory. Tried to allocate XX MiB`                            | 显存满                                        | 资源 ladder §3.1 步骤 1-4 依次往下                          |
| `bitsandbytes was unable to find the CUDA library`                         | venv 装的 bitsandbytes vs SLURM 的 CUDA 不匹配 | `pip install --force-reinstall bitsandbytes` 在 SLURM module 加载后跑 |
| `RuntimeError: No trainable parameters found for RL update`               | PEFT 没成功包模型                              | 确认 `engine._is_peft` 在 init 后是 True                    |
| `NoneType has no attribute 'shape'` 在 `_completion_log_prob`              | prompt 截断后 completion 也被切空              | `max_completion_length` 太小,涨到 ≥256                     |
| `policy_loss: nan`                                                         | 梯度爆炸 / lr 太大                            | lr 降一档 (1e-5 → 5e-6)                                    |
| `policy_loss` 不变 (相邻 step 完全一样)                                    | 模型没在更新                                  | 看 trainable params,确认 PEFT 装上了                       |
| `local_files_only=True, no file found`                                    | HF cache 路径里没有 Qwen3-14B                  | `ls $HF_CACHE_DIR/hub` 看实际存在的 model name              |
| Loaded model 但 `mean_reward` 永远 0                                       | batch_id 里没有 `grpo_reward` 字段             | 确认 batch 用的是 `RewardFunctionNorm` (新版),而不是旧的    |

---

## 6. 一个 1 小时能做完的 "调参 mini-experiment"

如果你只有 1 个 GPU-小时,我推荐这么花:

```text
0:00-0:15   smoke test (上一节 §2),拿到基线 metrics + 显存峰值
0:15-0:25   sweep learning_rate ∈ {5e-6, 1e-5, 2e-5} 各 1 step,看 policy_loss
0:25-0:35   选最稳定的 lr,sweep lora_r ∈ {4, 8, 16} 各 1 step
0:35-0:50   用最佳 (lr, r) 跑 10 step on batch_1,确认 policy_loss 曲线收敛
0:50-1:00   保存 LoRA,记录 metrics,sbatch 时间到自动退
```

每次改完参数,**改 SLURM 的 env var** 而不是改 python 文件 (避免 git diff 混乱):

```bash
# 例如想试 lr=2e-5,在 SLURM 里加
export GRPO_LR=2e-5
```

⚠️ 我现在的 `rl_demo.py` override 只读 `GRPO_QUANTIZATION/N_SAMPLES/MAX_LENGTH/MODEL_ID/BATCH_ID/FULL_TRAIN`。要加 `GRPO_LR` 等更多 knob,在 `rl_demo.py` 的 `_build_rl_config()` 里依样画葫芦多加几行:

```python
if "GRPO_LR" in os.environ:
    cfg.learning_rate = float(os.environ["GRPO_LR"])
if "GRPO_LORA_R" in os.environ:
    cfg.lora_r = int(os.environ["GRPO_LORA_R"])
if "GRPO_LORA_ALPHA" in os.environ:
    cfg.lora_alpha = int(os.environ["GRPO_LORA_ALPHA"])
```

---

## 7. 一句话总结

40GB 卡 → **必须 4-bit 量化** → 模型 7GB,激活预算 25GB → seq=1024、n=4 都装得下。
跑步骤：**smoke test 15 分钟 → 资源 ladder 调到不 OOM → 学习 ladder 一次一个超参 sweep → 长跑前看 batch_2 的 holdout reward**。
所有 SLURM-side 开关都通过 `export GRPO_XXX=...` 设定,不动 python 代码,git diff 干净。
