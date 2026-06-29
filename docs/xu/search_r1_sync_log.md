# Search-R1 同步记录

这个文件记录 `Search-R1` 项目的固定路径和每次从 Mac 同步到南信大 A100 的状态。

## 固定路径

```text
项目名：Search-R1
GitHub：https://github.com/XUHIT/Search-R1
Mac 主仓库：/Users/xuhaoxuan/Desktop/deepseek_one_week/repos/Search-R1
A100 运行副本：/mnt/xu/xu_exp/Search-R1
远端机器：nuist-a100-via-windows
远端账号：xu
```

## 同步规则

默认方向：

```text
Mac 主仓库 -> A100 运行副本
```

Mac 负责：

```text
clone / 修改 / commit / push
```

A100 负责：

```text
训练 / 推理 / 评测 / 保存 outputs 和 checkpoints
```

同步时默认排除：

```text
.git
__pycache__
.pytest_cache
outputs
checkpoints
wandb
swanlog
```

A100 运行副本不包含 `.git`，不要在 A100 运行副本里做 GitHub push。

## 当前状态

```text
本地分支：main
本地提交：598e61b
本地状态：clean，main 与 origin/main 一致
A100 运行副本大小：5.3M
A100 运行副本 .git：不存在
最后检查时间：2026-06-24 17:31 CST
```

## 说明书产物

```text
Markdown 源文件：/Users/xuhaoxuan/Desktop/deepseek_one_week/docs/search_r1_manual_cn.md
PDF 文件：/Users/xuhaoxuan/Desktop/deepseek_one_week/output/pdf/search_r1_manual_cn.pdf
PDF 构建脚本：/Users/xuhaoxuan/Desktop/deepseek_one_week/tools/build_search_r1_manual_pdf.py
```

PDF 检查结果：

```text
页数：18
文本抽取：中文标题可读，无替换字符
渲染检查：18 页均已渲染为 PNG，封面、正文、代码页、末页抽查正常
覆盖内容：项目用途、代码结构、模型、数据、检索索引、训练配置、A100 落地路径、已知坑和第一次跑通建议
```

## 同步日志

### 2026-06-29 15:13 CST

启动 3B-Instruct GRPO 300-step 对照实验，checkpoint 保存频率改为每 50 step。

```text
目的：接续 PPO 300-step 实验后，单独启动 GRPO 对照；不重复 PPO。验证 GRPO 在 n_agent=3、NQ-256 validation、FSDP offload=false 下的稳定性和 EM 曲线。
模型：/mnt/xu/xu_data/Search-R1/models/Qwen2.5-3B-Instruct
起点：原始 Instruct 模型，不接 PPO checkpoint。
```

运行配置：

```text
RUN_ID：20260629_grpo300_n3_save50_offfalse_070408_grpo
实验名：phase6-grpo-qwen2.5-3b-instruct-300step-n3-nq256-offfalse-save50-safe-20260629_grpo300_n3_save50_offfalse_070408
tmux：searchr1_grpo_20260629_grpo300_n3_save50_offfalse_070408
W&B run id：68dm0ekv
W&B URL：https://wandb.ai/xuhaoxuan-harbin-institute-of-technology/search-r1/runs/68dm0ekv
queue log：/mnt/xu/xu_exp/Search-R1/logs/queue_20260629_grpo300_n3_save50_offfalse_070408.log
train log：/mnt/xu/xu_exp/Search-R1/logs/phase6-grpo-qwen2.5-3b-instruct-300step-n3-nq256-offfalse-save50-safe-20260629_grpo300_n3_save50_offfalse_070408.log
outer log：/mnt/xu/xu_exp/Search-R1/logs/phase6-grpo-qwen2.5-3b-instruct-300step-n3-nq256-offfalse-save50-safe-20260629_grpo300_n3_save50_offfalse_070408.outer.log

TOTAL_STEPS=300
MAX_TURNS=3
TRAIN_BATCH_SIZE=512
VAL_BATCH_SIZE=256
VAL_SOURCE_DATA=/data/Search-R1/datasets/nq_hotpotqa_train/test.parquet
VAL_DATA_SOURCE_FILTER=nq
VAL_DATA_NUM=256
ALGORITHM_ADV_ESTIMATOR=grpo
ROLLOUT_N_AGENT=3
ACTOR_USE_KL_LOSS=true
ACTOR_KL_LOSS_COEF=0.001
ACTOR_KL_LOSS_TYPE=low_var_kl
PPO_MINI_BATCH_SIZE=256
PPO_MICRO_BATCH_SIZE=32
LOG_PROB_MICRO_BATCH_SIZE=64
REF_LOG_PROB_MICRO_BATCH_SIZE=64
ROLLOUT_GPU_MEMORY_UTILIZATION=0.35
MAX_RESPONSE_LENGTH=500
MAX_OBS_LENGTH=500
MAX_PROMPT_LENGTH=4096
MAX_START_LENGTH=2048
FSDP_OFFLOAD=false
SAVE_FREQ=50
TEST_FREQ=50
RETRIEVER_READY_TIMEOUT_SECONDS=1200
```

启动与健康检查：

```text
启动前 resource_check_ok：/mnt_free=6818GiB，blocked=20，iowait=14%，swap_out=0KiB/s，gpu_used_max=0MiB。
Flat GPU retriever 首次加载 61GB index 较慢；上次 600s timeout 失败，本次 timeout 放宽到 1200s 后 ready。
初始验证：step 0 val/test_score/nq=0.171875。
step 1 已完成，无 OOM；timing_s/step=255.476。
step 1 train score=0.255，reward=0.255，finish_ratio=0.975，ratio_of_valid_action=0.976，number_of_valid_search=0.986。
step 1 显存峰值观测约 28.6GB，8 卡利用率约 98%。
```

### 2026-06-28 00:50 CST

启动 3B-Instruct v0.2-like 500-step 训练，降低 vLLM 显存占比并使用完整 NQ 训练中验证。

```text
目的：在严格 v0.2 batch/micro-batch 形状基础上，把 MAX_TURNS 改为 3、vLLM gpu_memory_utilization 改为 0.4，训练 500 step；每 100 step 记录一次完整 NQ EM。
模型：/mnt/xu/xu_data/Search-R1/models/Qwen2.5-3B-Instruct
起点：原始 Instruct 模型，不接 checkpoint。
```

运行配置：

```text
RUN_ID：20260628_v02m3_gpu04_500_nqfull_004150
实验名：phase5-v02-qwen2.5-3b-instruct-500step-3turn-b512-gpu04-nqfull-20260628_v02m3_gpu04_500_nqfull_004150
tmux：searchr1_phase5_500_20260628_v02m3_gpu04_500_nqfull_004150
W&B run id：p7bhvtq1
W&B URL：https://wandb.ai/xuhaoxuan-harbin-institute-of-technology/search-r1/runs/p7bhvtq1
train log：/mnt/xu/xu_exp/Search-R1/logs/phase5-v02-qwen2.5-3b-instruct-500step-3turn-b512-gpu04-nqfull-20260628_v02m3_gpu04_500_nqfull_004150.outer.log
GPU 采样日志：/mnt/xu/xu_exp/Search-R1/logs/gpu_peak_phase5-v02-qwen2.5-3b-instruct-500step-3turn-b512-gpu04-nqfull-20260628_v02m3_gpu04_500_nqfull_004150.log

TOTAL_STEPS=500
MAX_TURNS=3
TRAIN_BATCH_SIZE=512
VAL_BATCH_SIZE=256
VAL_SOURCE_DATA=/data/Search-R1/datasets/nq_hotpotqa_train/test.parquet
VAL_DATA_SOURCE_FILTER=nq
VAL_DATA_NUM=null
PPO_MINI_BATCH_SIZE=256
PPO_MICRO_BATCH_SIZE=64
LOG_PROB_MICRO_BATCH_SIZE=128
REF_LOG_PROB_MICRO_BATCH_SIZE=128
CRITIC_PPO_MINI_BATCH_SIZE=256
CRITIC_PPO_MICRO_BATCH_SIZE=8
ACTOR_LR_WARMUP_RATIO=0.285
CRITIC_LR_WARMUP_RATIO=0.015
ROLLOUT_GPU_MEMORY_UTILIZATION=0.4
MAX_RESPONSE_LENGTH=500
MAX_OBS_LENGTH=500
MAX_PROMPT_LENGTH=4096
MAX_START_LENGTH=2048
FSDP_OFFLOAD=false
VAL_BEFORE_TRAIN=true
SAVE_FREQ=100
TEST_FREQ=100
```

启动确认：

```text
启动前：GPU 0-7 均为 0MiB，无运行中的 Search-R1 容器。
vmstat：blocked=20，iowait=15%，swap_out=0KiB/s。
完整 NQ test：3610 行；当前 DataLoader 整 batch 有效评估 256*14=3584 条。
配置已打印生效：gpu_memory_utilization=0.4，max_turns=3，val_batch_size=256，val_data_num=None，Size of val dataloader=14，Total training steps=500。
当前状态：正在执行 VAL_BEFORE_TRAIN 的完整 NQ 初始验证。
```

预计耗时：

```text
完整 NQ validation 预计 40-80 分钟一次。
本 run 有初始验证 + step 100/200/300/400/500，共 6 次完整 NQ validation。
训练 step 参考已有 b512 smoke，预计约 4-6 分钟/step。
总耗时粗估 38-58 小时；更保守按 2 天左右观察。
等初始验证完成并跑出前 1-2 个训练 step 后，可用实际 step time 重新估算。
```

### 2026-06-28 00:35 CST

完成严格 3B v0.2 shape、但 `FSDP_OFFLOAD=false` 的 3-step smoke。

```text
目的：复刻 v0.2 PPO 的关键 shape，用 smoke 判断 3B-Instruct 在 8xA100 40GB 上关闭 offload 是否会爆显存。
模型：/mnt/xu/xu_data/Search-R1/models/Qwen2.5-3B-Instruct
起点：原始 Instruct 模型，不接 checkpoint。
```

脚本改动：

```text
文件：
  /Users/xuhaoxuan/Desktop/deepseek_one_week/repos/Search-R1/scripts/xu/run_phase1_flat_ppo_safe_docker.sh
  /Users/xuhaoxuan/Desktop/deepseek_one_week/repos/Search-R1/scripts/xu/train_phase1_flat_ppo_wandb.sh

已同步到：
  /mnt/xu/xu_exp/Search-R1/scripts/xu/

新增并验证生效：
  ACTOR_LR_WARMUP_RATIO
  CRITIC_LR_WARMUP_RATIO
  ROLLOUT_GPU_MEMORY_UTILIZATION

原因：
  v0.2 的 actor warmup=0.285、critic warmup=0.015、vLLM gpu_memory_utilization=0.6；
  之前 xu wrapper 仍写死 0.0 / 0.0 / 0.35，严格 v0.2 smoke 必须透传。
```

运行配置：

```text
RUN_ID：20260628_v02strict3b_offfalse_smoke_retry_001252
实验名：phase4-v02strict-qwen2.5-3b-instruct-3step-4turn-b512-offfalse-20260628_v02strict3b_offfalse_smoke_retry_001252
tmux：searchr1_v02strict3b_smoke_20260628_v02strict3b_offfalse_smoke_retry_001252
W&B run id：h92mbma8
W&B URL：https://wandb.ai/xuhaoxuan-harbin-institute-of-technology/search-r1/runs/h92mbma8
train log：/mnt/xu/xu_exp/Search-R1/logs/phase4-v02strict-qwen2.5-3b-instruct-3step-4turn-b512-offfalse-20260628_v02strict3b_offfalse_smoke_retry_001252.outer.log
GPU 采样日志：/mnt/xu/xu_exp/Search-R1/logs/gpu_peak_phase4-v02strict-qwen2.5-3b-instruct-3step-4turn-b512-offfalse-20260628_v02strict3b_offfalse_smoke_retry_001252.log

TOTAL_STEPS=3
MAX_TURNS=4
TRAIN_BATCH_SIZE=512
VAL_BATCH_SIZE=256
VAL_DATA_NUM=256
PPO_MINI_BATCH_SIZE=256
PPO_MICRO_BATCH_SIZE=64
LOG_PROB_MICRO_BATCH_SIZE=128
REF_LOG_PROB_MICRO_BATCH_SIZE=128
CRITIC_PPO_MINI_BATCH_SIZE=256
CRITIC_PPO_MICRO_BATCH_SIZE=8
ACTOR_LR_WARMUP_RATIO=0.285
CRITIC_LR_WARMUP_RATIO=0.015
ROLLOUT_GPU_MEMORY_UTILIZATION=0.6
MAX_RESPONSE_LENGTH=500
MAX_OBS_LENGTH=500
MAX_PROMPT_LENGTH=4096
MAX_START_LENGTH=2048
FSDP_OFFLOAD=false
VAL_BEFORE_TRAIN=true
SAVE_FREQ=-1
TEST_FREQ=3
RAY_TMPFS_SIZE=32g
```

结果：

```text
状态：训练容器 Exited(0)，retriever 已清理，GPU 0-7 显存释放为 0MiB。
W&B：已同步。

启动前：
  /mnt_free=7427GiB
  blocked=20
  iowait=13%
  swap_out=0KiB/s
  gpu_used_max=0MiB

验证指标：
  Initial validation：nq=0.2109375，hotpotqa=0.28125
  Final validation：nq=0.203125，hotpotqa=0.28125

显存峰值：
  GPU0 37968MiB / 37.08GiB / 92.7%
  GPU1 37940MiB / 37.05GiB / 92.6%
  GPU2 37514MiB / 36.63GiB / 91.6%
  GPU3 37826MiB / 36.94GiB / 92.3%
  GPU4 38958MiB / 38.04GiB / 95.1%
  GPU5 37672MiB / 36.79GiB / 92.0%
  GPU6 39890MiB / 38.96GiB / 97.4%
  GPU7 38250MiB / 37.35GiB / 93.4%

结论：
  严格 3B v0.2 shape 且 offload=false 可以跑完 3-step smoke，没有 OOM。
  但 GPU6 峰值达到 97.4%，显存余量很薄。
  正式长训不建议再增加 MAX_TURNS / MAX_RESPONSE_LENGTH / batch。
  如果长训不稳，优先降低 ROLLOUT_GPU_MEMORY_UTILIZATION；其次降 MAX_RESPONSE_LENGTH 或 MAX_TURNS。
```

### 2026-06-27 23:45 CST

完成 Qwen2.5-3B-Instruct 原文式 batch=512 的 3-step smoke。

```text
目的：验证 data.train_batch_size=512 是否可在当前 8xA100 40GB 上运行，同时不把 PPO micro batch 错误放大到 512。
模型：/mnt/xu/xu_data/Search-R1/models/Qwen2.5-3B-Instruct
起点：原始 Instruct 模型，不接 checkpoint。
```

脚本改动：

```text
文件：
  /Users/xuhaoxuan/Desktop/deepseek_one_week/repos/Search-R1/scripts/xu/run_phase1_flat_ppo_safe_docker.sh
  /Users/xuhaoxuan/Desktop/deepseek_one_week/repos/Search-R1/scripts/xu/train_phase1_flat_ppo_wandb.sh

已同步到：
  /mnt/xu/xu_exp/Search-R1/scripts/xu/

改动：
  新增 PPO_MINI_BATCH_SIZE
  新增 PPO_MICRO_BATCH_SIZE
  新增 LOG_PROB_MICRO_BATCH_SIZE
  新增 REF_LOG_PROB_MICRO_BATCH_SIZE
  新增 CRITIC_PPO_MINI_BATCH_SIZE
  新增 CRITIC_PPO_MICRO_BATCH_SIZE

原因：
  原项目的 batch=512 是全局 rollout/update batch；
  不是把 actor/ref/logprob/critic micro batch 全部设成 512。
```

运行配置：

```text
RUN_ID：20260627_paperb512_smoke_232519
实验名：phase4-flat-ppo-qwen2.5-3b-instruct-3step-2turn-b512-paperbatch-safe-20260627_paperb512_smoke_232519
tmux：searchr1_paperb512_smoke_20260627_paperb512_smoke_232519
W&B run id：qlppbqlu
W&B URL：https://wandb.ai/xuhaoxuan-harbin-institute-of-technology/search-r1/runs/qlppbqlu
launcher log：/mnt/xu/xu_exp/Search-R1/logs/phase4-flat-ppo-qwen2.5-3b-instruct-3step-2turn-b512-paperbatch-safe-20260627_paperb512_smoke_232519.launcher.log
train log：/mnt/xu/xu_exp/Search-R1/logs/phase4-flat-ppo-qwen2.5-3b-instruct-3step-2turn-b512-paperbatch-safe-20260627_paperb512_smoke_232519.outer.log

TOTAL_STEPS=3
MAX_TURNS=2
TRAIN_BATCH_SIZE=512
VAL_BATCH_SIZE=256
VAL_DATA_NUM=256
PPO_MINI_BATCH_SIZE=256
PPO_MICRO_BATCH_SIZE=64
LOG_PROB_MICRO_BATCH_SIZE=128
REF_LOG_PROB_MICRO_BATCH_SIZE=128
CRITIC_PPO_MINI_BATCH_SIZE=512
CRITIC_PPO_MICRO_BATCH_SIZE=8
MAX_RESPONSE_LENGTH=256
MAX_OBS_LENGTH=500
MAX_PROMPT_LENGTH=4096
MAX_START_LENGTH=2048
FSDP_OFFLOAD=false
SAVE_FREQ=-1
TEST_FREQ=3
RAY_TMPFS_SIZE=32g
```

结果：

```text
状态：训练容器 Exited (0)，retriever 已清理，GPU 0-7 已释放为 0MiB。
W&B：已同步。

启动前：
  /mnt_free=7438GiB
  blocked=20
  iowait=12%
  swap_out=0KiB/s
  gpu_used_max=0MiB

显存观察：
  retriever ready 后：GPU0 6316MiB，GPU1-7 5840MiB
  vLLM/cache 初始化前后最高观察：GPU0 24834MiB，GPU1-7 约 24488-24508MiB
  step 1 rollout：约 14792-17404MiB
  step 1 update：最高观察约 20660MiB
  step 2 运行中最高观察：GPU0 25858MiB，其他卡约 24964-25628MiB
  结论：原文式 batch 拆分下，batch=512 未触发 OOM，观察峰值约 25.9GiB/卡。

step 1：
  critic/score/mean=0.217
  env/finish_ratio=0.871
  env/number_of_valid_search=1.082
  response_length/mean=710.867
  ACTIVE_TRAJ_NUM=[512, 467, 126, 66]
  timing_s/step=262.053

step 2：
  critic/score/mean=0.236
  env/finish_ratio=0.836
  env/number_of_valid_search=1.115
  response_length/mean=716.619
  ACTIVE_TRAJ_NUM=[512, 460, 165, 84]
  timing_s/step=246.671

final small validation at step 3：
  val/test_score/hotpotqa=0.203125
  val/test_score/nq=0.21875
```

说明：

```text
当前 TOTAL_STEPS=3 的训练循环实际记录 step 1 和 step 2 两个 PPO update，并在 step 3 做 final validation。
因此这个 run 只用于显存和稳定性 smoke，不作为训练效果结论。
```

### 2026-06-27 22:56 CST

启动 Qwen2.5-3B-Instruct 大 batch 阶段训练。

```text
目的：在阶段 2 有效的基础上提高 batch 和训练步数，继续观察 Search-R1 protocol NQ EM 是否提升。
模型：/mnt/xu/xu_data/Search-R1/models/Qwen2.5-3B-Instruct
起点：原始 Instruct 模型，不接任何已训练 checkpoint。

配置：
  TOTAL_STEPS=300
  MAX_TURNS=2
  TRAIN_BATCH_SIZE=64
  VAL_BATCH_SIZE=32
  MAX_RESPONSE_LENGTH=256
  MAX_OBS_LENGTH=500
  MAX_PROMPT_LENGTH=4096
  FSDP_OFFLOAD=false
  SAVE_FREQ=100
  TEST_FREQ=100
  RAY_TMPFS_SIZE=32g

RUN_ID：20260627_instruct300_t2_b64_144904
实验名：phase3-flat-ppo-qwen2.5-3b-instruct-300step-2turn-b64-safe-20260627_instruct300_t2_b64_144904
tmux：searchr1_instruct300_t2_b64_20260627_instruct300_t2_b64_144904
W&B run id：rg698os5
W&B URL：https://wandb.ai/xuhaoxuan-harbin-institute-of-technology/search-r1/runs/rg698os5
launcher log：/mnt/xu/xu_exp/Search-R1/logs/phase3-flat-ppo-qwen2.5-3b-instruct-300step-2turn-b64-safe-20260627_instruct300_t2_b64_144904.launcher.log
train log：/mnt/xu/xu_exp/Search-R1/logs/phase3-flat-ppo-qwen2.5-3b-instruct-300step-2turn-b64-safe-20260627_instruct300_t2_b64_144904.outer.log
```

启动前资源：

```text
/mnt_free=7446GiB
blocked=20
iowait=12%
swap_out=0KiB/s
gpu_used_max=0MiB
无运行中的 Search-R1 容器
```

首步观测：

```text
phase1 train rows：19200
Size of train dataloader：300
Total training steps：300
已进入 epoch 0, step 2
step 1 critic/score/mean：0.219
step 1 env/finish_ratio：0.797
step 1 env/number_of_valid_search：1.250
step 1 response_length/mean：744.984
step 1 timing_s/step：59.277
显存峰值观察：约 21-23GiB/卡，未 OOM
```

checkpoint 预期：

```text
由于已修复 PPO final checkpoint 保存逻辑，本 run 预期保存：
  actor/critic global_step_100
  actor/critic global_step_200
  actor/critic global_step_300
```

### 2026-06-27 23:01 CST

按用户要求手动停止 Qwen2.5-3B-Instruct 大 batch 阶段训练。

```text
实验名：phase3-flat-ppo-qwen2.5-3b-instruct-300step-2turn-b64-safe-20260627_instruct300_t2_b64_144904
停止前状态：训练容器 Up 8 minutes，retriever 容器 Up 12 minutes。
停止前 GPU：多数 GPU 约 31GiB，部分约 19GiB，训练已进入早期 step。
停止动作：只删除本实验训练容器和 searchr1-retriever-flat-safe，kill 对应 tmux；不碰其他用户进程。
停止后 GPU：0-7 显存均释放为 0MiB。
日志快照：/mnt/xu/xu_exp/Search-R1/logs/phase3-flat-ppo-qwen2.5-3b-instruct-300step-2turn-b64-safe-20260627_instruct300_t2_b64_144904.manual_stop_snapshot.log
checkpoint：未到 SAVE_FREQ=100，未形成可用训练 checkpoint。
```

### 2026-06-27 19:19 CST

启动两个训练后 checkpoint 的完整 Search-R1 protocol NQ EM 评估队列。

```text
目标：完整评估当前已落盘的两个 Qwen2.5-3B-Instruct 训练 checkpoint：
  1. 阶段 2 run 的 actor/global_step_50
  2. 上一轮 50-step run 的 actor/global_step_25

说明：阶段 2 run 没有 actor/global_step_100；上一轮 50-step run 没有 actor/global_step_50。因此只能评估各自实际保存的 checkpoint。
```

评估口径：

```text
协议：Search-R1 protocol
数据：/data/Search-R1/datasets/nq_hotpotqa_train/test.parquet
过滤：data_source=nq
NQ rows：3610
veRL val dataloader：112 batches，VAL_BATCH_SIZE=32，最后不满 batch 不计入时约 3584 条
检索：2018 Wikipedia dump，E5 Flat retriever，top-k=3
参数：MAX_TURNS=4，MAX_RESPONSE_LENGTH=500，MAX_OBS_LENGTH=500，MAX_PROMPT_LENGTH=4096，MAX_START_LENGTH=2048
指标：val/test_score/nq，Exact Match
模式：VAL_ONLY=true，VAL_BEFORE_TRAIN=true，SAVE_FREQ=-1
```

队列信息：

```text
PAIR_ID：20260627_fullnq_eval_pair2_111256
tmux：searchr1_fullnq_eval_pair2_20260627_fullnq_eval_pair2_111256
pair log：/mnt/xu/xu_exp/Search-R1/logs/20260627_fullnq_eval_pair2_111256.pair.launcher.log

第一个 eval：
  label：phase2_gs50_full_nq
  checkpoint：/mnt/xu/xu_exp/Search-R1/verl_checkpoints/phase2-flat-ppo-qwen2.5-3b-instruct-100step-2turn-safe-20260627_instruct100_t2_b8_091022/actor/global_step_50
  实验名：eval-phase2-instruct3b-gs50-nq-full-searchr1-20260627_phase2_gs50_fullnq_111256
  W&B run id：zeh97h1d
  W&B URL：https://wandb.ai/xuhaoxuan-harbin-institute-of-technology/search-r1/runs/zeh97h1d
  日志：/mnt/xu/xu_exp/Search-R1/logs/eval-phase2-instruct3b-gs50-nq-full-searchr1-20260627_phase2_gs50_fullnq_111256.outer.log
  状态：已启动，retriever ready，validation dataset size=3610，Size of val dataloader=112。

第二个 eval：
  label：phase1_gs25_full_nq
  checkpoint：/mnt/xu/xu_exp/Search-R1/verl_checkpoints/phase1-flat-ppo-qwen2.5-3b-instruct-50step-1turn-safe-20260627_instruct50_t1_b8_073233/actor/global_step_25
  状态：排队中，等待第一个 eval 完成后自动启动。
```

修正记录：

```text
第一次提交队列 20260627_fullnq_eval_pair_110416 在启动阶段失败，原因是 VAL_SOURCE_DATA 使用了主机路径 /mnt/xu/xu_data/Search-R1/...，容器内不可见。
已改为容器路径 /data/Search-R1/datasets/nq_hotpotqa_train/test.parquet 后重启。
```

### 2026-06-27 22:44 CST

完成两个训练后 checkpoint 的完整 Search-R1 protocol NQ EM 评估。

评估口径同 2026-06-27 19:19 CST 记录：

```text
数据：NQ test，data_source=nq，原始 3610 rows
veRL full-batch 计入：112 batches * 32 = 3584 条
检索：E5 Flat，2018 Wikipedia，top-k=3
参数：MAX_TURNS=4，MAX_RESPONSE_LENGTH=500，MAX_OBS_LENGTH=500，MAX_PROMPT_LENGTH=4096，MAX_START_LENGTH=2048
指标：val/test_score/nq，Exact Match
```

结果：

```text
阶段 2 checkpoint：
  checkpoint：/mnt/xu/xu_exp/Search-R1/verl_checkpoints/phase2-flat-ppo-qwen2.5-3b-instruct-100step-2turn-safe-20260627_instruct100_t2_b8_091022/actor/global_step_50
  实验名：eval-phase2-instruct3b-gs50-nq-full-searchr1-20260627_phase2_gs50_fullnq_111256
  W&B run id：zeh97h1d
  W&B URL：https://wandb.ai/xuhaoxuan-harbin-institute-of-technology/search-r1/runs/zeh97h1d
  日志：/mnt/xu/xu_exp/Search-R1/logs/eval-phase2-instruct3b-gs50-nq-full-searchr1-20260627_phase2_gs50_fullnq_111256.outer.log
  NQ EM：0.3150111607142857
  等价计数：约 1129 / 3584
  状态：容器 Exited(0)，W&B 已同步。

上一轮 50-step checkpoint：
  checkpoint：/mnt/xu/xu_exp/Search-R1/verl_checkpoints/phase1-flat-ppo-qwen2.5-3b-instruct-50step-1turn-safe-20260627_instruct50_t1_b8_073233/actor/global_step_25
  实验名：eval-phase1-instruct3b-gs25-nq-full-searchr1-20260627_phase1_gs25_fullnq_123736
  W&B run id：63rw6sjy
  W&B URL：https://wandb.ai/xuhaoxuan-harbin-institute-of-technology/search-r1/runs/63rw6sjy
  日志：/mnt/xu/xu_exp/Search-R1/logs/eval-phase1-instruct3b-gs25-nq-full-searchr1-20260627_phase1_gs25_fullnq_123736.outer.log
  NQ EM：0.25558035714285715
  等价计数：约 916 / 3584
  状态：容器 Exited(0)，W&B 已同步。
```

对比：

```text
Qwen2.5-3B-Instruct zero-shot Search-R1 protocol：0.21233258928571427，约 761 / 3584
上一轮 50-step run 的 actor/global_step_25：0.25558035714285715，约 916 / 3584，较 zero-shot +0.04325，约 +155 / 3584
阶段 2 run 的 actor/global_step_50：0.3150111607142857，约 1129 / 3584，较 zero-shot +0.10268，约 +368 / 3584
阶段 2 gs50 较上一轮 gs25：+0.05943，约 +213 / 3584
```

清理状态：

```text
GPU 0-7：0MiB
有效 eval 容器：均 Exited(0)
失败 run eval-phase2-instruct3b-gs50-nq-full-searchr1-20260627_phase2_gs50_fullnq_110416 是路径错误启动失败，已忽略。
```

### 2026-06-27 19:00 CST

修复 PPO 训练结束时不保存 final checkpoint 的问题。

```text
问题：Search-R1 当前 PPO 训练入口使用 verl/trainer/ppo/ray_trainer.py。该训练循环只在 self.global_steps % save_freq == 0 时保存 checkpoint；到达 total_training_steps 后只做 final validation 然后 return，没有 final save。因此阶段 2 run 在 step50 保存了 global_step_50，但 step100 final validation 后没有保存 global_step_100。

修改文件：
  Mac：/Users/xuhaoxuan/Desktop/deepseek_one_week/repos/Search-R1/verl/trainer/ppo/ray_trainer.py
  A100：/mnt/xu/xu_exp/Search-R1/verl/trainer/ppo/ray_trainer.py

修改内容：在 PPO final validation 后、return 前，若 trainer.save_freq > 0 且 actor/global_step_${global_steps} 不存在，则调用 self._save_checkpoint() 保存最终 actor/critic checkpoint。

验证：
  Mac：python3 -m py_compile verl/trainer/ppo/ray_trainer.py
  A100：python3 -m py_compile /mnt/xu/xu_exp/Search-R1/verl/trainer/ppo/ray_trainer.py
  A100 关键行：Saving final checkpoint at global_step_${self.global_steps}

影响：只影响后续新训练。已经结束的阶段 2 run 无法事后恢复 global_step_100 权重；当前仍只能评估 actor/global_step_50。
```

### 2026-06-27 18:53 CST

修复 Mac -> Windows -> A100 SSH 访问方式，新增稳定 tunnel alias。

```text
问题：原 alias nuist-a100-via-windows 通过 ProxyJump windows-bridge 访问 A100。windows-bridge 使用 link-local IPv6 地址 fe80::716b:c1c1:1c39:fde3%en7。网络重连后，Mac 端对 link-local 作用域路由偶发失败，表现为 No route to host 或 Operation not permitted。

修复：保留原 alias 不动，新增：
  windows-bridge-a100-tunnel
  nuist-a100-via-windows-tunnel

用法：
  ssh -fN windows-bridge-a100-tunnel
  ssh nuist-a100-via-windows-tunnel

原理：先建立一条持久 Mac -> Windows SSH LocalForward，把本机 localhost:10022 转发到 A100 10.255.251.134:22；后续 A100 命令只连接本机 10022，不再每次重复触发 ProxyJump/link-local 路由解析。

验证：
  ssh nuist-a100-via-windows-tunnel 'hostname; nvidia-smi'

验证结果：
  hostname=star
  /mnt free=7500G
  GPU 0-7 memory.used=0MiB

备份：
  /Users/xuhaoxuan/.ssh/config.bak-20260627-ssh-tunnel-fix
```

### 2026-06-27 17:18 CST

启动 Qwen2.5-3B-Instruct 阶段 2 Search-R1 PPO 训练。

```text
目的：从原始 Qwen2.5-3B-Instruct 重新开始训练，不接上一阶段 checkpoint；增加到 MAX_TURNS=2，让模型开始学习多轮 search -> information -> answer。
模型：/mnt/xu/xu_data/Search-R1/models/Qwen2.5-3B-Instruct
配置：TOTAL_STEPS=100，MAX_TURNS=2，TRAIN_BATCH_SIZE=8，VAL_BATCH_SIZE=32，MAX_RESPONSE_LENGTH=256，MAX_OBS_LENGTH=500，MAX_PROMPT_LENGTH=4096，FSDP_OFFLOAD=false，SAVE_FREQ=50，TEST_FREQ=50。
RUN_ID：20260627_instruct100_t2_b8_091022
实验名：phase2-flat-ppo-qwen2.5-3b-instruct-100step-2turn-safe-20260627_instruct100_t2_b8_091022
W&B run id：0q0jvdrx
W&B URL：https://wandb.ai/xuhaoxuan-harbin-institute-of-technology/search-r1/runs/0q0jvdrx
训练日志：/mnt/xu/xu_exp/Search-R1/logs/phase2-flat-ppo-qwen2.5-3b-instruct-100step-2turn-safe-20260627_instruct100_t2_b8_091022.outer.log
启动状态：资源检查通过，retriever ready，训练容器已启动并进入 PPO 训练循环。
启动前资源：/mnt_free=7546GiB，blocked=20，iowait=15%，swap_out=0KiB/s，gpu_used_max=0MiB。
第一步观测：env/finish_ratio=0.875，env/number_of_valid_search=1.125，response_length/mean=672.250，timing_s/step=54.632。
注意：按当前保存逻辑，SAVE_FREQ=50 很可能保存 global_step_50；最终 global_step_100 是否自动保存需等训练完成后确认。
```

### 2026-06-27 18:42 CST

完成 Qwen2.5-3B-Instruct 阶段 2 Search-R1 PPO 训练。

```text
RUN_ID：20260627_instruct100_t2_b8_091022
实验名：phase2-flat-ppo-qwen2.5-3b-instruct-100step-2turn-safe-20260627_instruct100_t2_b8_091022
W&B run id：0q0jvdrx
W&B URL：https://wandb.ai/xuhaoxuan-harbin-institute-of-technology/search-r1/runs/0q0jvdrx
训练容器状态：Exited(0)
retriever：已清理
GPU：0-7 显存均释放为 0MiB
W&B：已同步 5 个文件
```

训练内置小验证结果：

```text
验证集大小：32 rows（训练脚本默认 VAL_DATA_NUM=VAL_BATCH_SIZE=32）
step 50 小验证：val/test_score/nq=0.3125，val/test_score/hotpotqa=0.1875
step 100 最终小验证：val/test_score/nq=0.375，val/test_score/hotpotqa=0.4375
```

训练过程后段观测：

```text
step 90：critic/score/mean=0.750
step 91：critic/score/mean=0.250
step 92：critic/score/mean=0.500
step 93：critic/score/mean=0.250
step 94：critic/score/mean=0.375
step 95：critic/score/mean=0.375
step 96：critic/score/mean=0.500
step 97：critic/score/mean=0.250
step 98：critic/score/mean=0.500
step 99：critic/score/mean=0.250
step 99 timing_s/step=31.854
```

checkpoint 注意事项：

```text
已保存 actor/global_step_50 和 critic/global_step_50。
未发现 actor/global_step_100 或 critic/global_step_100。
因此当前可用于完整 NQ Search-R1 protocol 评估的阶段 2 checkpoint 是：
/mnt/xu/xu_exp/Search-R1/verl_checkpoints/phase2-flat-ppo-qwen2.5-3b-instruct-100step-2turn-safe-20260627_instruct100_t2_b8_091022/actor/global_step_50

step 100 的 0.375 是训练脚本内置 32 条小验证结果，不能等同于完整 NQ test EM。
```

### 2026-06-27 16:06 CST

完成 Qwen2.5-3B-Instruct 的 50-step Search-R1 PPO 训练验证。

```text
目标：按保守参数先做 50 个 PPO update，验证 Instruct 初始化在 MAX_TURNS=1、response=128 下能稳定训练并记录 W&B 曲线。
模型：/mnt/xu/xu_data/Search-R1/models/Qwen2.5-3B-Instruct
数据：/mnt/xu/xu_data/Search-R1/datasets/nq_hotpotqa_train
检索：Flat GPU retriever，/mnt/xu/xu_data/Search-R1/retrieval/wiki18_e5_flat
配置：TOTAL_STEPS=50，MAX_TURNS=1，TRAIN_BATCH_SIZE=8，VAL_BATCH_SIZE=32，MAX_RESPONSE_LENGTH=128，MAX_OBS_LENGTH=500，MAX_PROMPT_LENGTH=4096，FSDP_OFFLOAD=false，SAVE_FREQ=25，TEST_FREQ=25。
资源策略：不 kill 别人的进程；只管理自己的 phase1/retriever 容器；Ray /tmp/ray 使用 tmpfs；容器内 nice + ionice；训练完成后清理 retriever。
```

run 记录：

```text
RUN_ID：20260627_instruct50_t1_b8_073233
实验名：phase1-flat-ppo-qwen2.5-3b-instruct-50step-1turn-safe-20260627_instruct50_t1_b8_073233
W&B run id：igeg1dky
W&B URL：https://wandb.ai/xuhaoxuan-harbin-institute-of-technology/search-r1/runs/igeg1dky
训练日志：/mnt/xu/xu_exp/Search-R1/logs/phase1-flat-ppo-qwen2.5-3b-instruct-50step-1turn-safe-20260627_instruct50_t1_b8_073233.outer.log
Ray 日志：/mnt/xu/xu_exp/Search-R1/logs/ray_phase1-flat-ppo-qwen2.5-3b-instruct-50step-1turn-safe-20260627_instruct50_t1_b8_073233
checkpoint 目录：/mnt/xu/xu_exp/Search-R1/verl_checkpoints/phase1-flat-ppo-qwen2.5-3b-instruct-50step-1turn-safe-20260627_instruct50_t1_b8_073233
状态：训练容器 Exited(0)，retriever 已清理，GPU 0-7 显存释放为 0MiB，W&B 已同步。
```

关键结果：

```text
phase1 train rows：400
训练内置验证集大小：32 rows（VAL_DATA_NUM 默认等于 VAL_BATCH_SIZE=32）
step 25 小验证：val/test_score/nq=0.0625，val/test_score/hotpotqa=0.0625
step 50 最终小验证：val/test_score/nq=0.125，val/test_score/hotpotqa=0.0625
最后一批训练指标 step 49：critic/score/mean=0.125，env/finish_ratio=0.500，env/number_of_valid_search=0.750，timing_s/step=23.149
近期训练 reward：step42=0.250，step43=0.250，step44=0.000，step45=0.250，step46=0.250，step47=0.000，step48=0.250，step49=0.125
```

checkpoint 注意事项：

```text
已保存 actor/global_step_25 和 critic/global_step_25。
未发现 actor/global_step_50 或 critic/global_step_50。
因此当前可用于后续完整 NQ Search-R1 protocol 评估的训练后 checkpoint 是 global_step_25，不是 step 50。
step 50 的 0.125 是训练脚本内置 32 条小验证结果，不能等同于完整 NQ test EM。
```

### 2026-06-27 13:52 CST

完成 Qwen2.5-3B base / Instruct 的普通 RAG baseline NQ EM 评测。

RAG 数据构建：

```text
目标：给未经过 Search-R1 训练的 3B base / Instruct 补普通 RAG baseline。
数据来源：/mnt/xu/xu_data/Search-R1/datasets/nq_hotpotqa_train/test.parquet
过滤：data_source=nq，生成 NQ rows=3610
RAG 数据目录：/mnt/xu/xu_data/Search-R1/datasets/20260627_rag_nq_top3_ctx500_044454
RAG test parquet：/mnt/xu/xu_data/Search-R1/datasets/20260627_rag_nq_top3_ctx500_044454/test.parquet
构建方式：按 NQ question 调用 live E5 Flat retriever，top-k=3，然后把检索文档拼进 prompt 的 Context。
检索：2018 Wikipedia dump，E5 retriever，Flat GPU index
context 截断：按 Qwen2.5-3B tokenizer 截断到 500 tokens
prompt：沿用 scripts/data_process/nq_rag.py 的普通 RAG prompt 风格
注意：A100 上没有作者原始 retrieval cache，因此没有直接运行 nq_rag.py，而是用相同 retriever/corpus 现场检索一次后打包 parquet。
```

评估口径：

```text
评估入口：verl.trainer.main_ppo
模式：VAL_ONLY=true，VAL_BEFORE_TRAIN=true，do_search=false
实际 veRL full-batch 评测条数：3584（VAL_BATCH_SIZE=32，最后 26 条不满 batch 未计入）
指标：Exact Match / val/test_score/nq
生成参数：temperature=1.0，top_p=1.0，max_response_length=500，max_prompt_length=4096
训练：无
```

结果：

```text
Qwen2.5-3B base RAG：
  RUN_ID：20260627_base3b_rag_nq_top3ctx500_044953
  实验名：eval-qwen2.5-3b-base-nq-rag-top3ctx500-20260627_base3b_rag_nq_top3ctx500_044953
  W&B run id：idqts26r
  W&B URL：https://wandb.ai/xuhaoxuan-harbin-institute-of-technology/search-r1/runs/idqts26r
  日志：/mnt/xu/xu_exp/Search-R1/logs/eval-qwen2.5-3b-base-nq-rag-top3ctx500-20260627_base3b_rag_nq_top3ctx500_044953.outer.log
  NQ EM：0.05552455357142857
  等价计数：199 / 3584
  状态：容器 Exited(0)，W&B 已同步，GPU 0-7 显存释放为 0MiB。

Qwen2.5-3B-Instruct RAG：
  RUN_ID：20260627_instruct3b_rag_nq_top3ctx500_052615
  实验名：eval-qwen2.5-3b-instruct-nq-rag-top3ctx500-20260627_instruct3b_rag_nq_top3ctx500_052615
  W&B run id：n88gefe5
  W&B URL：https://wandb.ai/xuhaoxuan-harbin-institute-of-technology/search-r1/runs/n88gefe5
  日志：/mnt/xu/xu_exp/Search-R1/logs/eval-qwen2.5-3b-instruct-nq-rag-top3ctx500-20260627_instruct3b_rag_nq_top3ctx500_052615.outer.log
  NQ EM：0.29464285714285715
  等价计数：1056 / 3584
  状态：容器 Exited(0)，W&B 已同步，GPU 0-7 显存释放为 0MiB。
```

同 3B Search-R1 协议评估对比：

```text
Qwen2.5-3B base：
  Search-R1 protocol zero-shot：0.04408482142857143，约 158 / 3584
  Fixed RAG@3 ctx500：0.05552455357142857，约 199 / 3584
  RAG - Search-R1 protocol：+0.01143973214285714，约 +41 / 3584

Qwen2.5-3B-Instruct：
  Search-R1 protocol zero-shot：0.21233258928571427，约 761 / 3584
  Fixed RAG@3 ctx500：0.29464285714285715，约 1056 / 3584
  RAG - Search-R1 protocol：+0.08231026785714288，约 +295 / 3584
```

### 2026-06-27 11:50 CST

完成作者已训练 Search-R1 7B 模型在 NQ test 上的同口径 EM 评测。

```text
目标：按同一 Search-R1 原文测试参数评估作者训练好的 7B 模型。
模型：/mnt/xu/xu_data/Search-R1/models/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo
数据：/mnt/xu/xu_data/Search-R1/datasets/nq_hotpotqa_train/test.parquet
过滤：data_source=nq，生成 NQ rows=3610
实际 veRL full-batch 评测条数：3584（VAL_BATCH_SIZE=32，最后 26 条不满 batch 未计入）
指标：Exact Match / val/test_score/nq
检索：2018 Wikipedia dump，E5 retriever，top-k=3，Flat GPU index
推理参数：Search-R1 template，temperature=1.0，top_p=1.0，max_response_length=500，max_obs_length=500，max_prompt_length=4096，max_start_length=2048，max_turns=4
训练：无，VAL_ONLY=true，VAL_BEFORE_TRAIN=true
```

结果：

```text
RUN_ID：20260627_trained7b_nqem_full_001957
实验名：eval-searchr1-trained7b-nq-full-paper-20260627_trained7b_nqem_full_001957
W&B run id：f18u3dwn
W&B URL：https://wandb.ai/xuhaoxuan-harbin-institute-of-technology/search-r1/runs/f18u3dwn
日志：/mnt/xu/xu_exp/Search-R1/logs/eval-searchr1-trained7b-nq-full-paper-20260627_trained7b_nqem_full_001957.outer.log
NQ EM：0.4810267857142857
等价计数：1724 / 3584
状态：容器 Exited(0)，W&B 已同步，retriever 已清理，GPU 0-7 显存释放为 0MiB。
```

同口径参考：

```text
Qwen2.5-3B base：0.04408482142857143，约 158 / 3584，W&B run gus9zkv5。
Qwen2.5-3B-Instruct：0.21233258928571427，约 761 / 3584，W&B run 5rm26pkx。
Search-R1 trained 7B：0.4810267857142857，约 1724 / 3584，W&B run f18u3dwn。
注意：3B base/Instruct 和作者 7B 训练模型不是同 backbone；严格训练前后对比还应补测 Qwen2.5-7B 和 Qwen2.5-7B-Instruct。
```

### 2026-06-26 23:45 CST

完成 Qwen2.5-3B-Instruct 的 10-step Search-R1 PPO 对照实验。

```text
目标：验证 Instruct 初始化能否在当前 A100 压力下稳定跑通 10 step，并与 base 10-step 形成对照。
模型：/mnt/xu/xu_data/Search-R1/models/Qwen2.5-3B-Instruct
容器内模型路径：/data/Search-R1/models/Qwen2.5-3B-Instruct
数据：/mnt/xu/xu_data/Search-R1/datasets/nq_hotpotqa_train
检索：Flat GPU retriever，/mnt/xu/xu_data/Search-R1/retrieval/wiki18_e5_flat
配置：TOTAL_STEPS=10，MAX_TURNS=1，TRAIN_BATCH_SIZE=8，VAL_BATCH_SIZE=8，MAX_RESPONSE_LENGTH=64，MAX_OBS_LENGTH=512，FSDP_OFFLOAD=false，RAY_TMPFS_SIZE=32g。
资源策略：不 kill 别人的进程；放宽 b/iowait 门槛；要求 GPU 空闲、无实时 swap out。
```

启动器修复：

```text
文件：/Users/xuhaoxuan/Desktop/deepseek_one_week/repos/Search-R1/scripts/xu/run_phase1_flat_ppo_safe_docker.sh
A100：/mnt/xu/xu_exp/Search-R1/scripts/xu/run_phase1_flat_ppo_safe_docker.sh
修复：新增 MODEL_PATH 传入训练容器，并把主机路径 /mnt/xu/xu_data/Search-R1/... 自动映射为容器路径 /data/Search-R1/...。
原因：第一次 Instruct 启动把主机模型路径传进容器，训练脚本 test -d MODEL_PATH 静默失败，未进入 Ray/训练。
失败 run：20260626_instruct10step_pressure_152932，训练容器 Exited(1)，GPU 未进入训练占用。
```

成功 run：

```text
RUN_ID：20260626_instruct10step_pressure_153443
实验名：phase1-flat-ppo-qwen2.5-3b-instruct-10step-1turn-safe-20260626_instruct10step_pressure_153443
W&B run id：q7r1ykp4
W&B URL：https://wandb.ai/xuhaoxuan-harbin-institute-of-technology/search-r1/runs/q7r1ykp4
训练日志：/mnt/xu/xu_exp/Search-R1/logs/phase1-flat-ppo-qwen2.5-3b-instruct-10step-1turn-safe-20260626_instruct10step_pressure_153443.outer.log
Ray 日志：/mnt/xu/xu_exp/Search-R1/logs/ray_phase1-flat-ppo-qwen2.5-3b-instruct-10step-1turn-safe-20260626_instruct10step_pressure_153443
结果：训练容器 Exited(0)，retriever 已清理，8 张 A100 显存释放为 0MiB。
```

关键观测：

```text
确认实际加载 Instruct：日志中 _name_or_path=/data/Search-R1/models/Qwen2.5-3B-Instruct。
成功越过 epoch 0 step 2，并完成 step 1-9 与最终验证 step 10。
W&B API 验证：state=finished，num_rows=10，steps=[1,2,3,4,5,6,7,8,9,10]，has_actor_pg_loss=True，has_val=True。
最终验证：val/test_score/nq=0.000，val/test_score/hotpotqa=0.000。
step 2 样例：critic/score/mean=0.125，env/finish_ratio=0.125，timing_s/step=25.753。
step 7 样例：critic/score/mean=0.125，env/finish_ratio=0.250，timing_s/step=21.656。
```

### 2026-06-27 03:21 CST

完成未训练 Qwen2.5-3B base 在 NQ test 上的 Search-R1 原文测试参数 EM 评测。

```text
目标：记录/复刻原文口径下的原始基座模型 NQ EM。
模型：/mnt/xu/xu_data/Search-R1/models/Qwen2.5-3B
数据：/mnt/xu/xu_data/Search-R1/datasets/nq_hotpotqa_train/test.parquet
过滤：data_source=nq，生成 NQ rows=3610
实际 veRL full-batch 评测条数：3584（VAL_BATCH_SIZE=32，最后 26 条不满 batch 未计入）
指标：Exact Match / val/test_score/nq
检索：2018 Wikipedia dump，E5 retriever，top-k=3，Flat GPU index
推理参数：Search-R1 template，temperature=1.0，top_p=1.0，max_response_length=500，max_obs_length=500，max_prompt_length=4096，max_start_length=2048，max_turns=4
训练：无，VAL_ONLY=true，VAL_BEFORE_TRAIN=true
```

结果：

```text
RUN_ID：20260626_base_nqem_full_162348
实验名：eval-qwen2.5-3b-base-nq-full-paper-20260626_base_nqem_full_162348
W&B run id：gus9zkv5
W&B URL：https://wandb.ai/xuhaoxuan-harbin-institute-of-technology/search-r1/runs/gus9zkv5
日志：/mnt/xu/xu_exp/Search-R1/logs/eval-qwen2.5-3b-base-nq-full-paper-20260626_base_nqem_full_162348.outer.log
NQ EM：0.04408482142857143
等价计数：158 / 3584
状态：容器 Exited(0)，W&B state=finished，GPU 0-7 显存释放为 0MiB。
```

注意：

```text
此前 eval-qwen2.5-3b-base-nq-paper-20260626_base_nqem_paper_161414 只评了 32 条，EM=0.0625，不作为正式结果。
原因：脚本当时仍把 data.val_data_num 固定为 VAL_BATCH_SIZE；已修复为 data.val_data_num="${VAL_DATA_NUM}"。
```

### 2026-06-27 05:14 CST

完成未训练 Qwen2.5-3B-Instruct 在 NQ test 上的同口径 EM 评测，并形成 base vs instruct 对比。

```text
目标：与未训练 Qwen2.5-3B base 做同参数对照。
模型：/mnt/xu/xu_data/Search-R1/models/Qwen2.5-3B-Instruct
数据：/mnt/xu/xu_data/Search-R1/datasets/nq_hotpotqa_train/test.parquet
过滤：data_source=nq，生成 NQ rows=3610
实际 veRL full-batch 评测条数：3584（VAL_BATCH_SIZE=32，最后 26 条不满 batch 未计入）
指标：Exact Match / val/test_score/nq
检索：2018 Wikipedia dump，E5 retriever，top-k=3，Flat GPU index
推理参数：Search-R1 template，temperature=1.0，top_p=1.0，max_response_length=500，max_obs_length=500，max_prompt_length=4096，max_start_length=2048，max_turns=4
训练：无，VAL_ONLY=true，VAL_BEFORE_TRAIN=true
```

结果：

```text
RUN_ID：20260627_instruct_nqem_full_192310
实验名：eval-qwen2.5-3b-instruct-nq-full-paper-20260627_instruct_nqem_full_192310
W&B run id：5rm26pkx
W&B URL：https://wandb.ai/xuhaoxuan-harbin-institute-of-technology/search-r1/runs/5rm26pkx
日志：/mnt/xu/xu_exp/Search-R1/logs/eval-qwen2.5-3b-instruct-nq-full-paper-20260627_instruct_nqem_full_192310.outer.log
NQ EM：0.21233258928571427
等价计数：761 / 3584
状态：容器 Exited(0)，W&B state=finished，GPU 0-7 显存释放为 0MiB。
```

同口径对比：

```text
Qwen2.5-3B base：0.04408482142857143，约 158 / 3584，W&B run gus9zkv5。
Qwen2.5-3B-Instruct：0.21233258928571427，约 761 / 3584，W&B run 5rm26pkx。
Instruct - base：+0.16824776785714284，约 +603 / 3584。
```

### 2026-06-26 14:30 CST

完成 Search-R1 online W&B 凭据落地和不占 GPU 的最小上传测试。

```text
W&B 项目 URL：https://wandb.ai/xuhaoxuan-harbin-institute-of-technology/search-r1
WANDB_ENTITY：xuhaoxuan-harbin-institute-of-technology
trainer.project_name / W&B project：search-r1
本地凭据记录：/Users/xuhaoxuan/Desktop/deepseek_one_week/nuist_remote_bridge/secrets/credentials.local.md
A100 Docker env 文件：/mnt/xu/xu_exp/Search-R1/.secrets/wandb.env
A100 Docker env 权限：600
普通 README 和同步日志不记录 API key 本体
```

online 最小测试结果：

```text
测试方式：Docker 内 wandb.init + wandb.log 一个 connectivity 标量，不使用 GPU
结果：通过
run id：exlq01hp
run URL：https://wandb.ai/xuhaoxuan-harbin-institute-of-technology/search-r1/runs/exlq01hp
```

同步规则补充：

```text
Search-R1 .gitignore 已加入 **/.secrets、*.env、*.env.*。
后续 Mac -> A100 rsync --delete 同步代码时必须排除 .secrets 和 *.env，避免删除 A100 的 /mnt/xu/xu_exp/Search-R1/.secrets/wandb.env。
```

启动第一阶段 20-step 小实验：

```text
目标：Qwen2.5-3B + PPO + Flat GPU retriever + online W&B
本地脚本：/Users/xuhaoxuan/Desktop/deepseek_one_week/repos/Search-R1/scripts/xu/train_phase1_flat_ppo_wandb.sh
A100 脚本：/mnt/xu/xu_exp/Search-R1/scripts/xu/train_phase1_flat_ppo_wandb.sh
训练 tmux：searchr1_phase1_flat_ppo_wandb
retriever tmux：searchr1_retriever_flat_phase1
实验名：phase1-flat-ppo-qwen2.5-3b-20step-2turn-20260626_063612
W&B run：https://wandb.ai/xuhaoxuan-harbin-institute-of-technology/search-r1/runs/4ctwe3mp
训练日志：/mnt/xu/xu_exp/Search-R1/logs/phase1-flat-ppo-qwen2.5-3b-20step-2turn-20260626_063612.outer.log
retriever 日志：/mnt/xu/xu_exp/Search-R1/logs/retriever_flat_phase1_20260626_063312.log
已确认：retriever probe 通过，W&B online run 创建成功，训练进入 epoch 0 step 1，并完成 step 1 指标记录。
已观察：max_obs_length=256 偏小，出现 OBSERVATION TOO LONG；后续正式实验建议调大。
最终状态：失败退出，没有跑满 20 step。
退出位置：epoch 0 step 2 附近。
直接错误：Ray GCS 断连，日志出现 Failed to connect to GCS within 60 seconds / GCS may have been killed。
清理状态：训练容器已退出，retriever 容器已手动停止，8 张 A100 显存已释放到 0 MiB。
保留产物：W&B run 4ctwe3mp、本地 W&B 目录 /mnt/xu/xu_exp/Search-R1/wandb/run-20260626_063641-4ctwe3mp、训练日志 outer.log。
后续建议：下一次先做 2-3 step 稳定性复测，增大 max_obs_length，降低生成长度/turn 数，并检查 Ray 临时目录/GCS 日志后再跑 20-50 step。
```

3-step 保守复测结果：

```text
目标：验证修复 W&B tracking 后，Qwen2.5-3B + PPO + Flat GPU retriever 能否稳定跑过多个 step。
实验名：phase1-flat-ppo-qwen2.5-3b-3step-1turn-stable-20260626_073454
W&B run：https://wandb.ai/xuhaoxuan-harbin-institute-of-technology/search-r1/runs/m34sg39i
训练日志：/mnt/xu/xu_exp/Search-R1/logs/phase1-flat-ppo-qwen2.5-3b-3step-1turn-stable-20260626_073454.outer.log
Ray 日志：/mnt/xu/xu_exp/Search-R1/logs/ray_phase1-flat-ppo-qwen2.5-3b-3step-1turn-stable-20260626_073454
retriever 日志：/mnt/xu/xu_exp/Search-R1/logs/retriever_flat_phase1_stable_20260626_072810.log
配置：TOTAL_STEPS=3，MAX_TURNS=1，TRAIN_BATCH_SIZE=8，MAX_RESPONSE_LENGTH=64，MAX_OBS_LENGTH=512。
结果：完成 step 1；进入 epoch 0 step 2 后失败。
step 1 指标样例：actor/pg_loss=-0.415，critic/vf_loss=5.380，env/finish_ratio=0.125，timing_s/step=65.650。
W&B 实测：手动 wandb sync 后，run m34sg39i 从 running 变为 finished，但 history_keys_count 仍为 3，training_keys_count 仍为 0。
判断：真实训练异常退出时训练指标仍只进 stdout；后续要继续查 veRL 训练主路径的 W&B log/flush 调用。
直接错误：Worker PID 9178 无法连接 Ray GCS，随后 ActorDiedError。
Ray 现场：gcs_server.out 记录 node health check Deadline Exceeded、worker SYSTEM_ERROR；raylet.out 记录 worker connection error code 2 / End of file。
系统状态：无内核 OOM killer / NVIDIA Xid 记录；停止后 8 张 A100 显存为 0 MiB。
负载判断：失败时/失败后 A100 load average 极高，stop 后仍约 107 / 288 / 199，swap 几乎占满。这更像 Ray/GCS 在高 CPU/I/O/内存压力下失联，不像单纯 CUDA OOM。
清理状态：训练容器已停止，retriever 容器已停止，GPU 已释放；失败容器保留为 Exited(1) 便于后续 inspect。
下一步：不要在当前高负载状态继续重复同一配置；等负载下降后再跑 2-3 step，或进一步降低 Ray/训练压力后复测。
```

W&B Charts 问题定位与修复：

```text
现象：run 4ctwe3mp 的 W&B 页面 Charts 只有 System 面板。
日志事实：W&B Logs 里能看到 step:1 的 actor/critic/env 指标。
API 检查：run 4ctwe3mp 的 historyKeys 只有 3 个，training_keys_count=0，summary 中没有 actor/critic。
判断：训练指标只进入 stdout，没有进入 W&B history，所以 Charts 无法自动显示训练曲线。
修复文件：/Users/xuhaoxuan/Desktop/deepseek_one_week/repos/Search-R1/verl/utils/tracking.py
A100 已同步：/mnt/xu/xu_exp/Search-R1/verl/utils/tracking.py
修复内容：Tracking.log 对 wandb/mlflow backend 过滤并转换数值标量后再 log。
验证方式：不占 GPU 的 tracking-sanitize-check 测试。
验证 run：https://wandb.ai/xuhaoxuan-harbin-institute-of-technology/search-r1/runs/0buw20gt
验证结果：W&B history/Charts 能出现 actor/pg_loss、critic/vf_loss、env/finish_ratio。
注意：真实训练 run m34sg39i 在异常退出后手动 sync 仍没有 actor/critic/env history；说明 tracking-sanitize-check 的修复只能证明 W&B 后端可写，不能证明当前训练主路径在崩溃场景下一定 flush 成功。
```

### 2026-06-26 14:15 CST

复查 Search-R1 的 W&B 可用性，准备后续 20-50 step 小实验前的实验监控方案。

```text
Windows 跳板：可连接
A100：可连接，hostname 为 star
3090/A20：可连接，hostname 为 xu
```

W&B 检查结论：

```text
项目底层 tracking 入口：verl/utils/tracking.py
当前支持 backend：wandb / mlflow / console
当前不直接支持：swanlab
训练 Docker：searchr1:cu121-vllm063-flashattn
容器内 wandb 版本：0.26.1
A100 主机访问 https://wandb.ai：200
A100 主机访问 https://api.wandb.ai/graphql：405，说明 API 端点可达
容器访问 https://wandb.ai：200
容器访问 https://api.wandb.ai/graphql：405，说明容器内 API 端点可达
WANDB_API_KEY：未设置
wandb status：api_key 为 null
```

当前判断：

```text
W&B 包、代码入口、离线记录、网络连通性都没问题。
offline W&B + veRL Tracking 最小测试已通过。
离线测试产物：/mnt/xu/xu_exp/Search-R1/wandb_check/wandb/offline-run-20260626_060841-j1ll6wn1/run-j1ll6wn1.wandb
online W&B 还不能算验证通过，因为 A100/容器没有 W&B 登录态或 API key。
后续如果要网页实时监控，必须先提供 WANDB_API_KEY 或在 A100/容器内完成 wandb login。
没有 online 登录前，训练优先使用 trainer.logger=['console'] 或 WANDB_MODE=offline。
W&B 项目 URL：https://wandb.ai/xuhaoxuan-harbin-institute-of-technology/search-r1
WANDB_ENTITY：xuhaoxuan-harbin-institute-of-technology
trainer.project_name / W&B project：search-r1
online 运行时不要把 WANDB_API_KEY 写入 README；用环境变量、--env-file 或 wandb login 注入。
```

### 2026-06-26 01:35 CST

开始迁移 Search-R1 retriever Docker 镜像到 A100。

```text
镜像标签：searchr1-retriever:cu121-faiss18
3090 镜像 ID：0cf22e536363
3090 tar：/mnt/data/xu/projects/Search-R1-docker/searchr1_retriever_cu121_faiss18.tar
SHA256：b9e146edd88503a6ddb18bb29d5737ea6854d9fafd770069c1f6f81550023da7
Dockerfile：/mnt/data/xu/projects/Search-R1-docker/docker/retriever/Dockerfile
内容边界：只包含 retriever 环境和 Search-R1 代码，不包含 wiki-18 corpus/index/model
```

3090 已验证：

```text
torch 2.4.0，CUDA 12.1
容器内识别到 NVIDIA GeForce RTX 3090
faiss 1.8.0，StandardGpuResources=True，faiss.get_num_gpus()=1
实际跑过 FAISS GPU search
pyserini LuceneSearcher 可导入
Search-R1 DenseRetriever 可导入
```

迁移策略：

```text
最初尝试单文件 rsync + zstd，但速度仍受 cpolar 链路限制。
随后在 3090 上把 17G tar 切成 17 个约 1G 分片。
A100 通过 4 条 rsync 连接并行拉取分片。
拉完后自动逐片 sha256 校验、合并 tar、总 SHA256 校验、docker load、容器内最小验证。
```

当前任务：

```text
3090 分片目录：/mnt/data/xu/projects/Search-R1-docker/parts_retriever_cu121_faiss18
A100 分片目录：/mnt/xu/docker_imports/parts_retriever_cu121_faiss18
A100 目标 tar：/mnt/xu/docker_imports/searchr1_retriever_cu121_faiss18.tar
A100 tmux：rsync_searchr1_retriever_parallel
A100 日志：/mnt/xu/docker_imports/searchr1_retriever_parallel_load_20260626.log
本地脚本：/Users/xuhaoxuan/Desktop/deepseek_one_week/remote_scripts/import_searchr1_retriever_docker_parallel_20260626.sh
A100 脚本：/mnt/xu/docker_imports/import_searchr1_retriever_docker_parallel_20260626.sh
状态：已完成
完成结果：17 个分片校验通过，合并 tar 的总 SHA256 与预期一致，docker load 成功
A100 镜像：searchr1-retriever:cu121-faiss18，image id 145e5898c821，约 36.6GB
验证结果：torch 2.4.0 / CUDA 可用 / 8 张 GPU 可见 / faiss 1.8.0 / pyserini LuceneSearcher / DenseRetriever 均通过
```

补充运行脚本：

```text
启动 Flat GPU retriever：
本地：/Users/xuhaoxuan/Desktop/deepseek_one_week/repos/Search-R1/scripts/xu/start_retriever_flat_gpu.sh
A100：/mnt/xu/xu_exp/Search-R1/scripts/xu/start_retriever_flat_gpu.sh

启动 HNSW64 CPU/ANN retriever：
本地：/Users/xuhaoxuan/Desktop/deepseek_one_week/repos/Search-R1/scripts/xu/start_retriever_hnsw_cpu.sh
A100：/mnt/xu/xu_exp/Search-R1/scripts/xu/start_retriever_hnsw_cpu.sh

真实 retriever 1-step 训练 smoke：
本地：/Users/xuhaoxuan/Desktop/deepseek_one_week/repos/Search-R1/scripts/xu/train_smoke_real_retriever.sh
A100：/mnt/xu/xu_exp/Search-R1/scripts/xu/train_smoke_real_retriever.sh
```

当前实验判断：

```text
训练 Docker 已可用，且 fake retriever 1-step PPO smoke train 已通过。
真实 Search-R1 retriever 已启动过，并且 /retrieve 验证通过。
Qwen2.5-3B + 真实 retriever + 8 卡 A100 的 1-step PPO smoke train 已通过。
e5_Flat.index 已合成，可用于真实 retriever 验证和训练好模型 infer；如果用 --faiss_gpu，会占用可见 GPU 显存。
HNSW64 已合成为 e5_HNSW64.index，CPU/ANN retriever endpoint 验证已通过。
第一轮真实检索训练 smoke 已完成。
第一阶段暂不使用 HNSW64，先只走已验证的 Flat GPU retriever。
第一轮正式小实验建议：Flat GPU retriever + Qwen2.5-3B + PPO、20-50 step、batch=64/128 起步、max_turns=2、topk=3。
```

### 2026-06-26 13:20 CST

完成训练好模型 + Flat GPU retriever 的 infer 轨迹观察。

```text
模型：/data/Search-R1/models/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo
retriever：Flat GPU e5
retriever URL：http://127.0.0.1:8000/retrieve
retriever 日志：/mnt/xu/xu_exp/Search-R1/logs/retriever_flat_gpu_20260626_050743.log
推理脚本：/mnt/xu/xu_exp/Search-R1/scripts/xu/run_infer_trained_flat.sh
推理 Python：/mnt/xu/xu_exp/Search-R1/scripts/xu/infer_trained_model.py
```

轨迹 1：

```text
日志：/mnt/xu/xu_exp/Search-R1/logs/infer_trained_flat_20260626_051503.outer.log
内部日志：/mnt/xu/xu_exp/Search-R1/logs/infer_trained_flat_20260626_051505.log
问题：Mike Barnett negotiated many contracts including which player that went on to become general manager of CSKA Moscow of the KHL?
结果：模型完成 search -> information -> answer，但最后误答 Wayne Gretzky。
关键观察：检索结果已经包含 Sergei Fedorov；错误来自模型读信息/归因失败，不是 retriever 或链路失败。
```

轨迹 2：

```text
日志：/mnt/xu/xu_exp/Search-R1/logs/infer_trained_flat_big_little_lies_20260626_051844.outer.log
内部日志：/mnt/xu/xu_exp/Search-R1/logs/infer_trained_flat_20260626_051846.log
问题：big little lies season 2 how many episodes?
结果：通过。
轨迹：模型先搜索 “big little lies season 2 how many episodes”，再搜索 “how many episodes in Big Little Lies season 2”，检索结果包含 “All seven episodes”，最终回答 <answer> seven </answer>。
关键观察：这是第一阶段复盘好的 search-reason-answer 行为样例。
```

运行后状态：

```text
searchr1-infer-trained-flat 已退出
searchr1-infer-trained-flat2 已退出
searchr1-retriever-flat 已停止
8 张 A100 显存均为 0 MiB
```

### 2026-06-26 11:24 CST

确认阶段边界：HNSW64 暂时归档，不作为第一阶段任务。

```text
第一阶段目标：使用已验证的 Flat GPU retriever 跑训练好模型 infer 和 Qwen2.5-3B 小步训练。
HNSW64 状态：index 已合成，endpoint 已验证，但训练 smoke 未作为通过结论。
HNSW64 用途：后续优化 GPU 显存占用时再启用。
当前决策：不要继续在第一阶段重跑 HNSW64 smoke，避免把主线拖偏。
```

### 2026-06-26 11:17 CST

完成 HNSW64 endpoint 验证，并按用户要求暂停训练 smoke。

```text
HNSW64 服务容器：searchr1-retriever-hnsw-test
临时端口：8001
index：/data/Search-R1/retrieval/wiki18_e5_hnsw64/e5_HNSW64.index
corpus：/data/Search-R1/retrieval/wiki18_e5_flat/wiki-18.jsonl
验证请求：http://127.0.0.1:8001/retrieve
验证结果：通过，Eiffel Tower 查询返回真实 wiki-18 文档
对比：8001 HNSW64 与 8000 Flat 的 Eiffel Tower 首条结果基本一致
```

训练 smoke 状态：

```text
脚本：/mnt/xu/xu_exp/Search-R1/scripts/xu/run_train_smoke_hnsw_20260626.sh
日志：/mnt/xu/xu_exp/Search-R1/logs/train_smoke_hnsw_retriever_20260626.log
失败备份：/mnt/xu/xu_exp/Search-R1/logs/train_smoke_hnsw_retriever_20260626.failed_with_flat.log
第一次：Flat + HNSW 同时运行时，训练进入 epoch 0 step 1 后 Ray worker/GCS 异常，不作为 HNSW endpoint 失败。
第二次：停掉 Flat 后重跑，HNSW retriever_ok 200，训练进入 epoch 0 step 1；用户要求暂停测试，因此停止并释放资源。
```

暂停后的状态：

```text
已停止：searchr1-train-smoke-hnsw
已停止：searchr1-retriever-hnsw-test
已停止：searchr1-retriever-flat
8000 / 8001 retriever 端口不再监听
8 张 A100 显存均为 0 MiB
没有删除 HNSW64 index 或 part_aa / part_ab 分片
```

### 2026-06-26 10:35 CST

完成 HNSW64 index 合成。

```text
目录：/mnt/xu/xu_data/Search-R1/retrieval/wiki18_e5_hnsw64
输入分片：
  part_aa，40G
  part_ab，31G
输出文件：e5_HNSW64.index，71G
SHA256 文件：e5_HNSW64.index.sha256
SHA256：9487d8c21fbef5b1e3dca70d06685ecf6a4a6728b48ff510f0a6bca4fc81449c
脚本：/mnt/xu/xu_data/Search-R1/scripts/merge_hnsw64_20260626.sh
日志：/mnt/xu/xu_data/Search-R1/logs/merge_hnsw64_20260626.log
tmux：searchr1_merge_hnsw64，已结束
开始时间：2026-06-26 10:19:50 CST
完成时间：2026-06-26 10:33:12 CST
总用时：约 13 分 22 秒
```

结果判断：

```text
合成成功。
正式 index 文件已生成。
SHA256 校验已生成。
part_aa / part_ab 暂时保留，等 HNSW64 CPU/ANN retriever 启动验证通过后再删除，预计可回收约 71G。
```

### 2026-06-26 10:13 CST

完成真实 retriever + Search-R1 训练 Docker 的 1-step smoke test。

```text
测试目标：验证真实 wiki-18 Flat GPU retriever 能被 Search-R1 训练流程调用
retriever 启动脚本：/mnt/xu/xu_exp/Search-R1/scripts/xu/start_retriever_flat_gpu.sh
训练启动脚本：/mnt/xu/xu_exp/Search-R1/scripts/xu/train_smoke_real_retriever.sh
训练日志：/mnt/xu/xu_exp/Search-R1/logs/train_smoke_real_retriever_20260626_020704.log
retriever 镜像：searchr1-retriever:cu121-faiss18
训练镜像：searchr1:cu121-vllm063-flashattn
retriever index：/data/Search-R1/retrieval/wiki18_e5_flat/e5_Flat.index
retriever corpus：/data/Search-R1/retrieval/wiki18_e5_flat/wiki-18.jsonl
模型：/data/Search-R1/models/Qwen2.5-3B
数据：/data/Search-R1/datasets/nq_hotpotqa_train/train.parquet 抽样 8 条
训练步数：1 step
结果：通过
```

已验证：

```text
Flat GPU retriever 成功启动并监听 0.0.0.0:8000
GET / 返回 404 属于正常现象，因为 FastAPI root 没定义
POST /retrieve 返回 200，能返回真实 wiki-18 检索文本
训练容器能通过 http://127.0.0.1:8000/retrieve 调用真实 retriever
Search-R1 的 <search> -> /retrieve -> <information> -> reward -> PPO update 链路跑通
critic update 和 actor update 完成
训练日志打印 step:1 指标
最终验证指标打印 val/test_score/nq: 0.25 和 val/test_score/hotpotqa: 0.0
```

当前运行状态：

```text
训练 smoke 已结束。
Flat GPU retriever 仍在运行：
  Docker container: searchr1-retriever-flat
  tmux session: searchr1_retriever_flat
  GPU 显存占用：GPU0 约 6316 MiB，GPU1-7 各约 5840 MiB
```

注意：

```text
Ray metrics exporter 相关 ERROR 不影响训练结果，是 Ray 指标导出器没有成功启动。
日志里的 OBSERVATION TOO LONG 表示 max_obs_length=128 会截断检索结果；正式实验建议提高到 256/512 或使用脚本默认 500。
Flat GPU retriever 会常驻占用每张 A100 约 5.8-6.3G 显存；正式训练更推荐合成 HNSW64 后用 CPU ANN retriever。
```

### 2026-06-25 22:45 CST

完成 Search-R1 训练 Docker 的 1-step smoke test。

```text
测试目标：只验证训练环境能不能真的训练，不验证真实检索环境
A100 脚本：/mnt/xu/xu_exp/Search-R1/scripts/xu/train_smoke_fake_retriever.sh
本地脚本：/Users/xuhaoxuan/Desktop/deepseek_one_week/repos/Search-R1/scripts/xu/train_smoke_fake_retriever.sh
日志：/mnt/xu/xu_exp/Search-R1/logs/train_smoke_fake_retriever_20260625_143844.log
Docker 镜像：searchr1:cu121-vllm063-flashattn
模型：/data/Search-R1/models/Qwen2.5-3B
数据：/data/Search-R1/datasets/nq_hotpotqa_train/train.parquet 抽样 8 条
检索：fake retriever，只返回符合 Search-R1 格式的本地假结果
训练步数：1 step
结果：通过
```

已验证：

```text
Ray 初始化正常
8 张 A100 可用于训练
Qwen2.5-3B actor / critic / ref 可加载
vLLM rollout 可构建
Search-R1 search loop 可调用 retriever 并把 information 塞回上下文
critic update 和 actor update 完成
训练日志打印 step:1 指标
最终验证指标打印 val/test_score/nq 和 val/test_score/hotpotqa
```

限制：

```text
这不是 retriever 环境验收。
正式推理和正式训练前，仍要单独启动真实 retriever，并验证 wiki-18 corpus/index 可用。
```

### 2026-06-25 13:08 CST

补充 3090/A20 资源准备规则。

```text
3090/A20 上先压缩或打包资源目录。
如果 cpolar 单 TCP 流明显限速，再切片并行传输。
传输前后必须保留 sha256 校验。
A100 收到后先校验，再解压、合并、docker load 或放入资源目录。
```

当前 3090/A20 已准备干净 Search-R1 资源副本：

```text
/mnt/data/xu/projects/Search-R1
```

该副本来自 3090 上已有的 `/mnt/data/xu/projects/Search-R1-docker` 本地仓库，因为 3090 直连 GitHub 克隆时 TLS 连接被中断。

训练好模型下载任务已在 3090/A20 后台启动：

```text
tmux 会话：searchr1_good_model_download
来源：hf-mirror.com / PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo
3090 目标目录：/mnt/data/xu/searchr1_resources/models/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo
后续 A100 目标目录：/mnt/xu/xu_data/Search-R1/models/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo
日志：/mnt/data/xu/searchr1_resources/logs/download_good_model_searchr1_20260625.log
```

核心资源下载任务也已在 3090/A20 后台启动：

```text
tmux 会话：searchr1_core_resources_download
来源：hf-mirror.com
目标根目录：/mnt/data/xu/searchr1_resources
日志：/mnt/data/xu/searchr1_resources/logs/download_core_resources_searchr1_20260625.log
包含：e5-base-v2、wiki-18 corpus、wiki-18 e5 index、nq_hotpotqa_train、Qwen2.5-3B
处理：wiki-18.jsonl.gz 解压为 wiki-18.jsonl；part_aa/part_ab 合并为 e5_Flat.index；各目录生成 SHA256SUMS.txt
```

补充测速和策略修正：

```text
A100 直连 hf-mirror.com 样本速度约 7.8MB/s。
3090/A20 经 cpolar 到 A100 当前约 0.6-0.8MB/s。
因此公开资源最快路径不是 3090 -> A100，而是 hf-mirror.com -> A100。
```

A100 直连下载任务已启动：

```text
tmux 会话：searchr1_a100_direct_resources
A100 目标根目录：/mnt/xu/xu_data/Search-R1
日志：/mnt/xu/xu_data/Search-R1/logs/download_resources_from_hf_mirror_20260625.log
包含：训练好模型、e5-base-v2、Qwen2.5-3B、nq_hotpotqa_train、wiki-18 corpus、wiki-18 e5 index
处理：A100 上下载 part_aa/part_ab 后合并为 e5_Flat.index，随后删除 part_aa/part_ab，避免重复占用。
```

3090/A20 同时开始生成兜底资源包：

```text
tmux 会话：searchr1_package_runtime
包目录：/mnt/data/xu/searchr1_resources/packages
策略：只打运行必需资源；检索包包含 e5_Flat.index 和 wiki-18.jsonl.gz，不包含 part_aa/part_ab，避免重复传输。
```

第二阶段下载队列已在 A100 创建：

```text
tmux 会话：searchr1_a100_stage2_resources
脚本：/mnt/xu/xu_data/Search-R1/scripts/download_stage2_resources_from_hf_mirror_20260625.sh
日志：/mnt/xu/xu_data/Search-R1/logs/download_stage2_resources_from_hf_mirror_20260625.log
状态：等待 searchr1_a100_direct_resources 完成后自动开始
```

队列包含：

```text
Qwen/Qwen2.5-7B
Qwen/Qwen2.5-3B-Instruct
Qwen/Qwen2.5-7B-Instruct
PeterJinGo/wiki-18-e5-index-HNSW64，合并为 e5_HNSW64.index
```

BM25 暂不默认下载，因为它是另一路稀疏检索方案，依赖 pyserini / Java / Lucene。当前先准备 HNSW64 作为第二阶段检索对照。

### 2026-06-25 11:26 CST

开始把 3090 上构建好的 Search-R1 Docker 镜像传到 A100。

```text
来源服务器：XU_a20_cpolar / 7.tcp.vip.cpolar.cn:10966
来源文件：/mnt/data/xu/projects/Search-R1-docker/searchr1_cu121_vllm063_flashattn.tar
来源大小：15,561,673,216 bytes，约 15G
镜像标签：searchr1:cu121-vllm063-flashattn
A100 目标文件：/mnt/xu/docker_imports/searchr1_cu121_vllm063_flashattn.tar
A100 Docker data-root：/mnt/xu/docker
传输方式：A100 通过专用 SSH key 直接 rsync 拉取 3090 文件
A100 tmux 会话：rsync_searchr1_docker
传输日志：/mnt/xu/docker_imports/searchr1_rsync_20260625.log
当前状态：传输完成，A100 已导入并通过最小测试
```

说明：

```text
最初尝试 scp -3 经 Mac 中转，约 1 分钟只传 19M，速度太慢。
随后在 A100 生成专用 SSH key，并把公钥加入 3090 authorized_keys。
当前使用 rsync --append-verify，可利用已传残片并支持断点续传。
```

### 2026-06-25 21:55 CST

完成 Search-R1 Docker 镜像导入和最小可用性验证。

```text
A100 镜像标签：searchr1:cu121-vllm063-flashattn
A100 tar：/mnt/xu/docker_imports/searchr1_cu121_vllm063_flashattn.tar
Docker image ID：e4d7eb512ca2
Docker image size：约 31.3GB
GPU 验证：容器内 nvidia-smi -L 可见 8 张 NVIDIA A100-SXM4-40GB
Python 验证：torch 2.4.0+cu121, cuda_available=True, cuda_device_count=8
依赖验证：vllm / flash_attn / verl / wandb 均可导入
挂载验证：/mnt/xu/xu_exp/Search-R1 -> /workspace/Search-R1，/mnt/xu/xu_data/Search-R1 -> /data/Search-R1 可用
```

标准进入方式已写入工作区总 README：

```text
/Users/xuhaoxuan/Desktop/deepseek_one_week/README.md
```

检查进度：

```bash
ssh nuist-a100-via-windows 'tmux capture-pane -pt rsync_searchr1_docker -S -40'
ssh nuist-a100-via-windows 'tail -n 20 /mnt/xu/docker_imports/searchr1_rsync_20260625.log'
ssh nuist-a100-via-windows 'ls -lh /mnt/xu/docker_imports/searchr1_cu121_vllm063_flashattn.tar'
```

### 2026-06-24 17:31 CST

完成 Search-R1 中文说明书。

```text
源文件：docs/search_r1_manual_cn.md
PDF：output/pdf/search_r1_manual_cn.pdf
构建脚本：tools/build_search_r1_manual_pdf.py
结果：成功
验证：pdfinfo、pdfplumber 文本抽取、pdftoppm 渲染 PNG、人工抽查关键页面
```

### 2026-06-24 16:55 CST

首次同步。

```text
来源：/Users/xuhaoxuan/Desktop/deepseek_one_week/repos/Search-R1
目标：nuist-a100-via-windows:/mnt/xu/xu_exp/Search-R1
提交：598e61b
结果：成功
远端大小：5.3M
远端 .git：不存在，符合运行副本规则
```

同步命令：

```bash
rsync -av --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude 'outputs' \
  --exclude 'checkpoints' \
  --exclude 'wandb' \
  --exclude 'swanlog' \
  --exclude '.secrets' \
  --exclude '*.env' \
  repos/Search-R1/ \
  nuist-a100-via-windows:/mnt/xu/xu_exp/Search-R1/
```
