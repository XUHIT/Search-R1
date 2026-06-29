# deepseek_one_week 工作区总说明

这个 README 记录当前工作区的通用约定。以后只要没有特别说明，Codex 默认按这里的规则协助管理项目、同步代码、连接远程服务器和运行实验。

## 默认机器分工

当前默认分工：

```text
Mac 本地
  - 运行 Codex
  - 管理 GitHub：clone / edit / commit / push
  - 保存研究笔记、方案文档、轻量输出
  - 通过 Windows/ENAgent 跳到南信大 A100

Windows 桥接机
  - 运行 ENAgent / 南信大 VPN
  - 作为 Mac 到南信大服务器的 SSH 跳板
  - 不作为主要代码仓库或长期数据存储

南信大 A100 服务器
  - 运行训练、推理、评测等实验
  - 保存数据集、conda 环境、实验输出和 checkpoints
```

当前默认远程目标：

```bash
ssh nuist-a100-via-windows
```

除非明确要求，不默认使用 4090 服务器。

更稳定的 A100 访问方式：

```bash
ssh -fN windows-bridge-a100-tunnel
ssh nuist-a100-via-windows-tunnel
```

用途：长时间训练、完整评估、频繁查日志时优先使用这个 tunnel alias。

原因：原来的 `nuist-a100-via-windows` 每次命令都会通过 `ProxyJump windows-bridge` 新建一条 Mac -> Windows -> A100 的链路。Windows 跳板使用的是 link-local IPv6 地址 `fe80::716b:c1c1:1c39:fde3%en7`，重插网线/重连网络后，Mac 端对这个作用域地址的路由偶尔会抖，表现为 `No route to host` 或 `Operation not permitted`。新的方式先用一条持久 SSH 隧道把本机 `localhost:10022` 转发到 A100 的 `10.255.251.134:22`，之后所有 A100 SSH 都只连本机端口，不再每条命令重复触发 `ProxyJump` 和 link-local 路由解析。

如果 tunnel 失效，重新执行：

```bash
ssh -fN windows-bridge-a100-tunnel
```

验证：

```bash
ssh nuist-a100-via-windows-tunnel 'hostname; nvidia-smi'
```

当前 SSH 配置备份：

```text
/Users/xuhaoxuan/.ssh/config.bak-20260627-ssh-tunnel-fix
```

可用的本地/备用服务器入口：

```sshconfig
Host XU_a20_cpolar
  HostName 7.tcp.vip.cpolar.cn
  User xu
  Port 10966
  IdentityFile ~/.ssh/id_rsa
```

使用方式：

```bash
ssh XU_a20_cpolar
```

这个入口通过 `cpolar` 公网转发访问本地服务器，可用于临时同步、检查或本地实验；它不是南信大校园网链路，也不是当前默认 A100 训练目标。

## GitHub 与代码同步规则

默认做法：

```text
Mac 是 GitHub 主仓库
A100 是运行副本
```

原因：Mac 访问 GitHub 更稳定，也更适合 commit/push；A100 主要负责跑实验，不依赖它直接访问 GitHub。

本地 GitHub 项目默认放：

```text
/Users/xuhaoxuan/Desktop/deepseek_one_week/repos
```

A100 运行副本默认放：

```text
/mnt/xu/xu_exp/项目名
```

常用流程：

```text
1. Mac 上 clone / 修改 / commit / push
2. Codex 把代码同步到 A100 的 /mnt/xu/xu_exp/项目名
3. A100 上运行实验
4. 大结果留在 A100，必要的小结果再同步回 Mac
```

同步代码时默认排除：

```text
.git
__pycache__
.pytest_cache
outputs
checkpoints
wandb
swanlog
.secrets
*.env
```

大文件、数据集、模型权重和 checkpoints 默认不进 GitHub。

## 当前项目映射

已同步项目：

```text
项目名：Search-R1
GitHub：https://github.com/XUHIT/Search-R1
Mac 主仓库：/Users/xuhaoxuan/Desktop/deepseek_one_week/repos/Search-R1
A100 运行副本：/mnt/xu/xu_exp/Search-R1
远端默认账号：xu
远端默认机器：nuist-a100-via-windows
当前本地提交：598e61b
```

`Search-R1` 的 GitHub 操作默认在 Mac 主仓库做；A100 上的 `/mnt/xu/xu_exp/Search-R1` 是运行副本，不包含 `.git`。

Search-R1 中文说明书：

```text
Markdown 源文件：/Users/xuhaoxuan/Desktop/deepseek_one_week/docs/search_r1_manual_cn.md
PDF 文件：/Users/xuhaoxuan/Desktop/deepseek_one_week/output/pdf/search_r1_manual_cn.pdf
PDF 构建脚本：/Users/xuhaoxuan/Desktop/deepseek_one_week/tools/build_search_r1_manual_pdf.py
```

说明书内容覆盖项目用途、代码结构、模型、数据、检索索引、训练配置、A100 落地路径、已知坑和第一次跑通建议。

Search-R1 学习路线：

```text
/Users/xuhaoxuan/Desktop/deepseek_one_week/docs/search_r1_learning_plan_cn.md
```

核心顺序：先看中文说明书建立项目地图，再粗读第一篇论文，再跑训练好的模型观察 search-reason-answer 轨迹，然后读核心代码、配环境、做小步数 smoke train，最后再正式训练未训练模型。

Search-R1 模型权重与数据资源清单：

```text
/Users/xuhaoxuan/Desktop/deepseek_one_week/docs/search_r1_resource_checklist_cn.md
```

Search-R1 资源准备的核心分工：

```text
3090/A20：资源准备机
  - 下载模型权重
  - 下载数据集、语料、检索索引
  - 做一次性预处理，例如解压 wiki-18、合并 e5 index、生成 parquet
  - 打包资源并计算大小/校验

A100：训练运行机
  - 保存最终资源到 /mnt/xu/xu_data/Search-R1
  - docker load / 运行 Search-R1 容器
  - 启动 retriever
  - 跑 infer.py、smoke train、正式训练

Mac/Codex：控制与记录机
  - 维护 GitHub、本地 README、同步日志
  - 发起/检查远端任务
  - 不作为 15G/60G+ 大资源的中转首选
```

原则：下载和一次性重处理尽量在 3090/A20 上做，A100 尽量只使用已经准备好的本地资源跑训练/推理。训练和推理阶段不要依赖 Hugging Face API 或外部检索 API。

3090/A20 上保留一份干净的 Search-R1 资源准备副本：

```text
/mnt/data/xu/projects/Search-R1
```

这份副本用于下载脚本、数据预处理、资源目录打包和校验，不用于正式训练。

补充测速结论：A100 直连 `hf-mirror.com` 的样本速度约 7.8MB/s，明显快于 3090/A20 经 cpolar 到 A100 的约 0.6-0.8MB/s。因此公开模型和公开数据资源优先让 A100 直接从 `hf-mirror.com` 下载到 `/mnt/xu/xu_data/Search-R1`；3090/A20 已下载资源主要作为备份、校验和兜底传输源。

同步日志：

```text
nuist_remote_bridge/search_r1_sync_log.md
```

本地实时监控页面：

```text
服务脚本：/Users/xuhaoxuan/Desktop/deepseek_one_week/tools/remote_gpu_monitor.py
页面文件：/Users/xuhaoxuan/Desktop/deepseek_one_week/tools/remote_gpu_monitor.html
当前本地地址：http://127.0.0.1:8765/
```

用途：在浏览器里实时观察 A100 和 3090 的 GPU 显存、GPU 利用率、CPU、I/O 等待、内存、磁盘、tmux 会话和主要进程。页面本身不直接 SSH；本地 Python 服务负责通过已有 SSH alias 查询远端。

## A100 服务器目录规则

A100 账号：

```text
host: star
user: xu
SSH alias: nuist-a100-via-windows
稳定 tunnel alias: nuist-a100-via-windows-tunnel
```

默认目录分工：

```text
代码/项目：/mnt/xu/xu_exp
数据集：/mnt/xu/xu_data
conda 环境：/mnt/xu/miniconda3
/home/xu：只放配置、小日志、SSH 等轻量文件
```

不要把项目、大数据、模型权重、长期输出放在 `/home/xu`。

进入 A100：

```bash
ssh nuist-a100-via-windows
```

长任务推荐：

```bash
ssh -fN windows-bridge-a100-tunnel
ssh nuist-a100-via-windows-tunnel
```

激活常用环境：

```bash
source /mnt/xu/miniconda3/etc/profile.d/conda.sh
conda activate EnerGen
```

查看 GPU：

```bash
nvidia-smi
```

A100 的 `nvidia-smi` 输出较慢，等 40-60 秒是正常的，不要用太短超时判断失败。

当前 A100 配置：

```text
8 x NVIDIA A100-SXM4-40GB
40960 MiB per GPU
Driver 535.183.01
CUDA 12.2
```

当前 A100 Docker 状态：

```text
Docker: 29.1.3
Docker data-root: /mnt/xu/docker
NVIDIA Container Toolkit: 1.19.1
Docker runtime: nvidia 已注册，默认 runtime 仍是 runc
xu 用户：已加入 docker 组，重新登录后可直接使用 docker
```

当前 Search-R1 Docker 镜像：

```text
镜像标签：searchr1:cu121-vllm063-flashattn
A100 tar：/mnt/xu/docker_imports/searchr1_cu121_vllm063_flashattn.tar
来源 tar：/mnt/data/xu/projects/Search-R1-docker/searchr1_cu121_vllm063_flashattn.tar
验证时间：2026-06-25
验证状态：已通过最小可用性测试
```

已通过的最小测试：

```text
容器能看到 8 张 A100
torch 2.4.0+cu121 可用，torch.cuda.device_count() = 8
vllm 可导入
flash_attn 可导入
verl 可导入
wandb 可导入
挂载 /mnt/xu/xu_exp/Search-R1 和 /mnt/xu/xu_data/Search-R1 后仍能正常进入项目
```

训练链路 smoke test：

```text
测试时间：2026-06-25
测试目标：只验证训练 Docker 能不能真的进入 Search-R1 训练循环，不验证真实检索环境
测试脚本：/mnt/xu/xu_exp/Search-R1/scripts/xu/train_smoke_fake_retriever.sh
本地脚本：/Users/xuhaoxuan/Desktop/deepseek_one_week/repos/Search-R1/scripts/xu/train_smoke_fake_retriever.sh
测试日志：/mnt/xu/xu_exp/Search-R1/logs/train_smoke_fake_retriever_20260625_143844.log
模型：/data/Search-R1/models/Qwen2.5-3B
数据：从 /data/Search-R1/datasets/nq_hotpotqa_train/train.parquet 抽 8 条样本
检索：本地 fake retriever，返回 Search-R1 期望的 JSON 结构，但不使用真实 wiki index
GPU：8 张 A100
训练步数：1 step
结果：通过
```

这个测试已经验证：

```text
Ray 能启动
Qwen2.5-3B actor / critic / ref 能加载
vLLM rollout 能构建
Search-R1 的 <search> -> retriever -> <information> -> reward -> PPO update 链路能跑通
critic update 和 actor update 都完成
最终打印 step:1 训练指标和 final validation metrics
```

这个测试没有验证：

```text
真实 retriever 环境
真实 wiki-18 e5 / HNSW64 / BM25 index 质量
训练好模型的真实推理效果
长时间训练稳定性
```

真实 retriever + 训练链路 smoke test：

```text
测试时间：2026-06-26
测试目标：验证真实 wiki-18 Flat GPU retriever 能被 Search-R1 训练流程调用
retriever 脚本：/mnt/xu/xu_exp/Search-R1/scripts/xu/start_retriever_flat_gpu.sh
训练脚本：/mnt/xu/xu_exp/Search-R1/scripts/xu/train_smoke_real_retriever.sh
训练日志：/mnt/xu/xu_exp/Search-R1/logs/train_smoke_real_retriever_20260626_020704.log
retriever：/data/Search-R1/retrieval/wiki18_e5_flat/e5_Flat.index + wiki-18.jsonl
retriever 镜像：searchr1-retriever:cu121-faiss18
训练镜像：searchr1:cu121-vllm063-flashattn
模型：/data/Search-R1/models/Qwen2.5-3B
数据：从 /data/Search-R1/datasets/nq_hotpotqa_train/train.parquet 抽 8 条样本
GPU：8 张 A100
训练步数：1 step
结果：通过
```

这个测试已经验证：

```text
Flat GPU retriever 可以启动并监听 0.0.0.0:8000
http://127.0.0.1:8000/retrieve 可以返回真实检索结果
训练容器可以通过 retriever.url 调用真实 retriever
真实 <search> -> /retrieve -> <information> -> reward -> PPO update 链路能跑通
critic update 和 actor update 都完成
最终打印 step:1 训练指标和 final validation metrics
```

注意事项：

```text
Flat GPU retriever 当前约占 GPU0 6.3G、GPU1-7 各 5.8G 显存。
smoke 日志里的 Ray metrics exporter ERROR 不影响训练，是指标导出器没起来，不是训练失败。
smoke 日志出现 OBSERVATION TOO LONG，说明 max_obs_length=128 会截断检索信息；正式实验建议提高到 256/512 或使用脚本默认 500。
这个测试只说明链路可跑，不代表训练效果已经可靠。
```

Search-R1 retriever Docker 镜像：

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

A100 迁移状态：

```text
启动时间：2026-06-26
目标 tar：/mnt/xu/docker_imports/searchr1_retriever_cu121_faiss18.tar
传输方式：3090 上切成 17 个约 1G 分片，A100 通过 4 条 rsync 连接并行拉取
A100 tmux：rsync_searchr1_retriever_parallel
A100 日志：/mnt/xu/docker_imports/searchr1_retriever_parallel_load_20260626.log
本地脚本：/Users/xuhaoxuan/Desktop/deepseek_one_week/remote_scripts/import_searchr1_retriever_docker_parallel_20260626.sh
A100 脚本：/mnt/xu/docker_imports/import_searchr1_retriever_docker_parallel_20260626.sh
完成时间：2026-06-26
状态：已完成，17 个分片校验通过，合并 tar 的总 SHA256 与预期一致，docker load 成功
A100 镜像：searchr1-retriever:cu121-faiss18，image id 145e5898c821，约 36.6GB
验证：torch 2.4.0 / CUDA 可用 / 8 张 GPU 可见 / faiss 1.8.0 / pyserini LuceneSearcher / DenseRetriever 均通过
```

retriever 镜像加载后使用时，要把 A100 上的资源目录挂进去：

```text
项目代码：/workspace/Search-R1
模型/语料/index：/data/Search-R1
```

Search-R1 当前可执行实验路线：

```text
已验证：
1. 训练 Docker 依赖可用
2. Qwen2.5-3B + fake retriever + 8 卡 A100 的 1-step PPO smoke train 可完成
3. Flat GPU retriever 可以启动，/retrieve 可以返回真实 wiki-18 检索结果
4. Qwen2.5-3B + 真实 retriever + 8 卡 A100 的 1-step PPO smoke train 可完成
5. HNSW64 CPU/ANN retriever 可以启动，8001 /retrieve 可以返回真实 wiki-18 检索结果
6. 训练好模型 SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo 可用 Flat GPU retriever 跑 infer 轨迹

下一步优先做：
1. 复盘训练好模型 infer 轨迹，明确什么是好的 search-reason-answer 行为
2. 使用已验证的 Flat GPU retriever 跑 Qwen2.5-3B + 真实检索的 20-50 step 小实验
3. 再考虑 GRPO 或 7B，不在第一阶段处理 HNSW64

不建议立刻做：
1. 直接完整 1005 step
2. 直接上 7B 正式训练
3. 继续在第一阶段纠缠 HNSW64 retriever
```

建议参数：

```text
真实检索 smoke train：
模型：/data/Search-R1/models/Qwen2.5-3B
数据：/data/Search-R1/datasets/nq_hotpotqa_train/train.parquet 抽 8 条
训练步数：1-2 step
train_batch_size：8
ppo_mini_batch_size：8
ppo_micro_batch_size：8
max_turns：1
retriever.topk：3
logger：console
save_freq / test_freq：-1
vllm gpu_memory_utilization：0.35

第一轮正式小实验：
模型：Qwen2.5-3B
方法：PPO
数据：nq_hotpotqa_train
训练步数：先 20-50 step，确认吞吐、显存、reward、日志和 checkpoint
train_batch_size：64 或 128 起步，不直接用 512
max_turns：2
retriever.topk：3
logger：offline wandb 或 console

第二轮实验：
模型：Qwen2.5-3B
方法：GRPO
注意：GRPO 默认 n_agent=5，会显著增加 rollout 开销；先小 batch、小 step

第三轮实验：
模型：Qwen2.5-7B
方法：PPO 或 GRPO
条件：3B PPO/GRPO 的真实检索 smoke 和小步训练稳定后再做
```

Search-R1 v0.2 PPO 参数核查：

```text
来源脚本：
/Users/xuhaoxuan/Desktop/deepseek_one_week/repos/Search-R1/scripts/nq_hotpotqa/v0.2/train_ppo.sh

总体结论：
v0.2 的 PPO 参数基本都在当前 Search-R1/veRL 代码路径里生效，不是只写在脚本里的摆设。
关键点是 data.train_batch_size=512 代表全局 rollout/update batch，不代表把所有 micro batch 设成 512。
正确复刻必须保留 actor/ref/logprob/critic 的 micro batch 拆分。
```

核心参数逐项结论：

```text
路径与运行：
  data_name=nq_hotpotqa_train：只影响 DATA_DIR 拼接；保持当前 nq_hotpotqa_train。
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7：生效；保持 8 卡。
  DATA_DIR=data/nq_hotpotqa_train：生效；容器内需映射为 /data/Search-R1/datasets/nq_hotpotqa_train 或由 wrapper 生成 phase data。
  WAND_PROJECT=Search-R1：生效但只影响 W&B project 名；我们可继续用 search-r1。
  BASE_MODEL=Qwen/Qwen2.5-7B：生效；当前 3B-Instruct 实验改为 /data/Search-R1/models/Qwen2.5-3B-Instruct。
  EXPERIMENT_NAME：生效；必须写清楚 3B-Instruct、v0.2-like、step/batch/turn。
  VLLM_ATTENTION_BACKEND=XFORMERS：生效；保持。
  PYTHONUNBUFFERED=1：生效；保持，方便实时日志。

数据：
  data.train_files：生效；正式训练用完整 train.parquet。
  data.val_files：生效；正式验证用 test.parquet。
  data.train_data_num=null：生效；null 表示不抽样，保持。
  data.val_data_num=null：生效；正式复刻用 null，smoke 可临时设小。
  data.train_batch_size=512：生效；可用，已用 3B-Instruct smoke 验证。
  data.val_batch_size=256：生效；正式 v0.2 口径保持。
  data.max_prompt_length=4096：生效；保持。
  data.max_response_length=500：生效；当前非 v0.2 smoke 用过 256，复刻需改回 500。
  data.max_start_length=2048：生效；保持。
  data.max_obs_length=500：生效；保持。
  data.shuffle_train_dataloader=True：生效；保持。

PPO / actor：
  algorithm.adv_estimator=gae：生效；选择 PPO+critic 路径，先保持。
  actor_rollout_ref.model.path：生效；当前用本地 3B-Instruct。
  actor_rollout_ref.actor.optim.lr=1e-6：生效；保持。
  actor_rollout_ref.model.enable_gradient_checkpointing=true：生效；保持。
  actor_rollout_ref.model.use_remove_padding=True：生效；保持。
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285：生效；当前 wrapper 已支持 ACTOR_LR_WARMUP_RATIO。
  actor_rollout_ref.actor.ppo_mini_batch_size=256：生效；保持。
  actor_rollout_ref.actor.ppo_micro_batch_size=64：生效；保持，不能跟 train_batch_size 绑定到 512。
  actor_rollout_ref.actor.fsdp_config.param_offload=true：生效；严格 v0.2 用 true，3B 为速度可先 false。
  actor_rollout_ref.actor.fsdp_config.grad_offload=true：生效；严格 v0.2 用 true，3B 为速度可先 false。
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true：生效；严格 v0.2 用 true，3B 为速度可先 false。
  actor_rollout_ref.actor.state_masking=true：生效；只训练非 <information> 区域，非常重要，保持。

rollout / ref：
  actor_rollout_ref.rollout.log_prob_micro_batch_size=128：生效；保持。
  actor_rollout_ref.rollout.tensor_model_parallel_size=1：生效；3B 保持 1。
  actor_rollout_ref.rollout.name=vllm：生效；保持。
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6：生效；当前 wrapper 已支持 ROLLOUT_GPU_MEMORY_UTILIZATION。
  actor_rollout_ref.ref.log_prob_micro_batch_size=128：生效；保持。
  actor_rollout_ref.ref.fsdp_config.param_offload=True：生效；严格 v0.2 用 true，3B 可先 false。
  actor_rollout_ref.rollout.n_agent=1：生效；PPO 保持 1。
  actor_rollout_ref.rollout.temperature=1：生效；保持。
  actor_rollout_ref.rollout.top_p=1.0：生效；保持。

critic：
  critic.optim.lr=1e-5：生效；保持。
  critic.model.use_remove_padding=True：生效；保持。
  critic.optim.lr_warmup_steps_ratio=0.015：生效；当前 wrapper 已支持 CRITIC_LR_WARMUP_RATIO。
  critic.model.path：生效；当前用本地 3B-Instruct。
  critic.model.enable_gradient_checkpointing=true：生效；保持。
  critic.ppo_micro_batch_size=8：生效；保持。
  critic.model.fsdp_config.param_offload=true：生效；严格 v0.2 用 true，3B 可先 false。
  critic.model.fsdp_config.grad_offload=true：生效；严格 v0.2 用 true，3B 可先 false。
  critic.model.fsdp_config.optimizer_offload=true：生效；严格 v0.2 用 true，3B 可先 false。

算法与训练器：
  algorithm.kl_ctrl.kl_coef=0.001：生效；保持。
  algorithm.no_think_rl=false：生效；必须保持 false，当前代码中 true 会走未完成分支。
  trainer.critic_warmup=0：生效；保持。
  trainer.logger=['wandb']：生效；我们建议 ['console','wandb']，console 日志作兜底。
  +trainer.val_only=false：生效；训练时保持 false。
  +trainer.val_before_train=true：生效；正式 v0.2-like 应打开。
  trainer.default_hdfs_dir=null：生效；保持 null。
  trainer.n_gpus_per_node=8：生效；保持。
  trainer.nnodes=1：生效；保持。
  trainer.save_freq=100：生效；保持。PPO final-save patch 已修复最终 checkpoint。
  trainer.test_freq=100：生效；保持，长训每 100 step 做验证。
  trainer.project_name：生效；当前 W&B project 用 search-r1。
  trainer.experiment_name：生效；每次实验必须唯一。
  trainer.total_epochs=15：生效，但 total_training_steps 非空时主要由 total_training_steps 控制。
  trainer.total_training_steps=1005：生效；正式复刻用 1005，资源验证阶段先 100/300。
  trainer.default_local_dir：生效；必须指向 /workspace/Search-R1/verl_checkpoints/<experiment>。

Search-R1 环境：
  max_turns=4：生效；v0.2 复刻需从当前短 smoke 的 2 改回 4。
  retriever.url=http://127.0.0.1:8000/retrieve：生效；保持本机 retriever。
  retriever.topk=3：生效；保持，和标准评估一致。
```

当前 wrapper 支持的 v0.2 batch / warmup / rollout 变量：

```text
已支持：
  TRAIN_BATCH_SIZE
  VAL_BATCH_SIZE
  PPO_MINI_BATCH_SIZE
  PPO_MICRO_BATCH_SIZE
  LOG_PROB_MICRO_BATCH_SIZE
  REF_LOG_PROB_MICRO_BATCH_SIZE
  CRITIC_PPO_MINI_BATCH_SIZE
  CRITIC_PPO_MICRO_BATCH_SIZE
  ACTOR_LR_WARMUP_RATIO=0.285
  CRITIC_LR_WARMUP_RATIO=0.015
  ROLLOUT_GPU_MEMORY_UTILIZATION=0.6
```

当前建议的 v0.2-like 3B-Instruct 路线：

```text
第一步：v0.2 shape smoke
  TOTAL_STEPS=3
  MAX_TURNS=4
  MAX_RESPONSE_LENGTH=500
  TRAIN_BATCH_SIZE=512
  VAL_BATCH_SIZE=256
  VAL_DATA_NUM=256
  PPO_MINI_BATCH_SIZE=256
  PPO_MICRO_BATCH_SIZE=64
  LOG_PROB_MICRO_BATCH_SIZE=128
  REF_LOG_PROB_MICRO_BATCH_SIZE=128
  CRITIC_PPO_MINI_BATCH_SIZE=256
  CRITIC_PPO_MICRO_BATCH_SIZE=8
  FSDP_OFFLOAD=false
  ROLLOUT_GPU_MEMORY_UTILIZATION=0.6

第一步 smoke 结果：
  run：phase4-v02strict-qwen2.5-3b-instruct-3step-4turn-b512-offfalse-20260628_v02strict3b_offfalse_smoke_retry_001252
  W&B：h92mbma8
  结果：训练容器 Exited(0)，retriever 已清理，8 张 A100 显存释放为 0MiB。
  初始验证：nq=0.2109375，hotpotqa=0.28125。
  最终验证：nq=0.203125，hotpotqa=0.28125。
  显存峰值：GPU6 39890MiB，约 38.96GiB/40GiB，97.4%。
  结论：严格 3B v0.2 shape 且 FSDP_OFFLOAD=false 可以跑完 3-step smoke，但显存余量很薄；正式长训建议保留显存保护，必要时优先降低 ROLLOUT_GPU_MEMORY_UTILIZATION 或缩短 MAX_RESPONSE_LENGTH/MAX_TURNS。

第二步：短训
  TOTAL_STEPS=100
  SAVE_FREQ=100
  TEST_FREQ=100
  VAL_BEFORE_TRAIN=true

第三步：中训
  TOTAL_STEPS=300
  SAVE_FREQ=100
  TEST_FREQ=100

第四步：原 v0.2 步数
  TOTAL_STEPS=1005
  SAVE_FREQ=100
  TEST_FREQ=100
  VAL_DATA_NUM=null
```

GRPO v0.2 差异：

```text
脚本：scripts/nq_hotpotqa/v0.2/train_grpo.sh
主要不同：
  algorithm.adv_estimator=grpo
  actor_rollout_ref.actor.use_kl_loss=true
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.rollout.n_agent=5

这些参数也生效，但 n_agent=5 会显著放大 rollout 和检索压力。
当前不建议直接上 GRPO v0.2；先把 PPO v0.2-like 在 3B-Instruct 上跑稳。
```

retriever 资源选择：

```text
e5_Flat.index：
已合成，约 61G。适合先做真实 retriever 功能验证和训练好模型推理验证。
如果加 --faiss_gpu，代码会用 faiss.index_cpu_to_all_gpus + shard + float16，把 index 分片放到可见 GPU 上。
训练同时运行时要注意它会占 GPU 显存。

e5_HNSW64.index：
已合成，约 71G。
适合 CPU ANN 检索，正式训练时更容易避免和训练进程抢 GPU 显存；但第一阶段暂不使用，先只走已验证的 Flat GPU retriever。
路径：/mnt/xu/xu_data/Search-R1/retrieval/wiki18_e5_hnsw64/e5_HNSW64.index
SHA256：9487d8c21fbef5b1e3dca70d06685ecf6a4a6728b48ff510f0a6bca4fc81449c
合成时间：2026-06-26 10:19:50-10:33:12 CST，总用时约 13 分 22 秒。
part_aa / part_ab 暂时保留，等后续确认不需要回滚再删除以回收约 71G。
HNSW64 endpoint 验证：已通过，临时端口 8001，Eiffel Tower 查询返回真实 wiki-18 文档。
HNSW64 训练 smoke：已启动并进入 epoch 0 step 1，随后按用户要求暂停；未作为通过结论。
```

已准备脚本：

```text
启动 Flat GPU retriever：
/mnt/xu/xu_exp/Search-R1/scripts/xu/start_retriever_flat_gpu.sh

启动 HNSW64 CPU/ANN retriever：
/mnt/xu/xu_exp/Search-R1/scripts/xu/start_retriever_hnsw_cpu.sh

真实 retriever 1-step 训练 smoke：
/mnt/xu/xu_exp/Search-R1/scripts/xu/train_smoke_real_retriever.sh

训练好模型 + Flat GPU retriever 推理：
/mnt/xu/xu_exp/Search-R1/scripts/xu/run_infer_trained_flat.sh
/mnt/xu/xu_exp/Search-R1/scripts/xu/infer_trained_model.py
```

训练好模型 infer 观察记录：

```text
时间：2026-06-26
模型：/data/Search-R1/models/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo
retriever：Flat GPU e5，http://127.0.0.1:8000/retrieve

轨迹 1：
日志：/mnt/xu/xu_exp/Search-R1/logs/infer_trained_flat_20260626_051503.outer.log
问题：Mike Barnett negotiated many contracts including which player that went on to become general manager of CSKA Moscow of the KHL?
现象：模型能搜索并检索到 Sergei Fedorov 文档，但最后误答 Wayne Gretzky。
结论：这是一条“链路通但推理读信息失败”的反例。

轨迹 2：
日志：/mnt/xu/xu_exp/Search-R1/logs/infer_trained_flat_big_little_lies_20260626_051844.outer.log
问题：big little lies season 2 how many episodes?
现象：模型两轮搜索，检索到 “All seven episodes”，最终回答 <answer> seven </answer>。
结论：这是第一阶段要复盘的好轨迹样例。

运行后状态：infer 容器和 Flat retriever 均已停止，8 张 A100 显存为 0 MiB。
```

GPU 容器建议启动参数：

```bash
docker run --gpus all --ipc=host --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 ...
```

Search-R1 Docker 最小验收命令：

```bash
docker image ls searchr1

docker run --rm --gpus all \
  searchr1:cu121-vllm063-flashattn \
  nvidia-smi -L

docker run --rm --gpus all \
  searchr1:cu121-vllm063-flashattn \
  python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count()); import vllm, flash_attn, verl, wandb; print('ok')"
```

以后每次使用 Search-R1 环境，优先这样进入容器：

```bash
ssh nuist-a100-via-windows
tmux new -s searchr1

docker run --rm -it \
  --name searchr1-dev \
  --gpus all \
  --ipc=host \
  --shm-size=64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /mnt/xu/xu_exp/Search-R1:/workspace/Search-R1 \
  -v /mnt/xu/xu_data/Search-R1:/data/Search-R1 \
  -w /workspace/Search-R1 \
  searchr1:cu121-vllm063-flashattn \
  bash
```

容器内固定路径：

```text
项目代码：/workspace/Search-R1
模型/数据/索引：/data/Search-R1
```

进入容器后，训练和推理都优先使用 `/data/Search-R1` 下的本地模型、数据和索引，不默认从 Hugging Face 直接下载。

Search-R1 实验监控与 W&B：

```text
项目代码支持 W&B。
训练脚本里常见设置：trainer.logger=['wandb']
底层入口：verl/utils/tracking.py
当前 Docker 镜像默认：WANDB_MODE=offline
当前 veRL tracking 支持：wandb / mlflow / console
当前 veRL tracking 不直接支持：swanlab
```

2026-06-25 测试结果：

```text
A100 主机可以连通 https://wandb.ai
A100 主机可以连通 https://api.wandb.ai
容器内 wandb 包可导入，版本 0.26.1
容器默认 WANDB_MODE=offline，因此默认不向 W&B 上传
```

2026-06-26 复查结果：

```text
Windows 跳板、A100、3090/A20 链路正常。
A100 主机访问 https://wandb.ai 返回 200。
A100 主机访问 https://api.wandb.ai/graphql 返回 405，说明 API 端点可达，GET 方法不适用。
训练 Docker 容器访问 https://wandb.ai 返回 200。
训练 Docker 容器访问 https://api.wandb.ai/graphql 返回 405，说明容器内也能到达 W&B API。
训练 Docker 中 wandb 版本：0.26.1。
训练 Docker 中 WANDB_API_KEY 未设置。
训练 Docker 中 wandb status 显示 api_key: null。
离线 W&B + veRL Tracking 最小测试已通过。
离线测试产物：/mnt/xu/xu_exp/Search-R1/wandb_check/wandb/offline-run-20260626_060841-j1ll6wn1/run-j1ll6wn1.wandb
online W&B 目前不能算通过，原因不是网络不可达，而是 A100/容器里还没有登录 W&B。
```

2026-06-26 online W&B 登录测试结果：

```text
完整 API key 已保存到本地 ignored 凭据文件和 A100 ignored env 文件。
普通 README 不记录 key 本体。
本地凭据记录：/Users/xuhaoxuan/Desktop/deepseek_one_week/nuist_remote_bridge/secrets/credentials.local.md
A100 Docker env 文件：/mnt/xu/xu_exp/Search-R1/.secrets/wandb.env
A100 Docker env 权限：600
online 最小测试：通过
测试 run：https://wandb.ai/xuhaoxuan-harbin-institute-of-technology/search-r1/runs/exlq01hp
```

2026-06-26 第一阶段 20-step 小实验：

```text
目标：验证 Qwen2.5-3B + PPO + Flat GPU retriever + online W&B 的真实训练链路
本地脚本：/Users/xuhaoxuan/Desktop/deepseek_one_week/repos/Search-R1/scripts/xu/train_phase1_flat_ppo_wandb.sh
A100 脚本：/mnt/xu/xu_exp/Search-R1/scripts/xu/train_phase1_flat_ppo_wandb.sh
训练 tmux：searchr1_phase1_flat_ppo_wandb
retriever tmux：searchr1_retriever_flat_phase1
实验名：phase1-flat-ppo-qwen2.5-3b-20step-2turn-20260626_063612
W&B run：https://wandb.ai/xuhaoxuan-harbin-institute-of-technology/search-r1/runs/4ctwe3mp
训练日志：/mnt/xu/xu_exp/Search-R1/logs/phase1-flat-ppo-qwen2.5-3b-20step-2turn-20260626_063612.outer.log
retriever 日志：/mnt/xu/xu_exp/Search-R1/logs/retriever_flat_phase1_20260626_063312.log
当前已确认：retriever probe 通过，W&B online run 创建成功，训练进入 epoch 0 step 1，并完成 step 1 指标记录。
已观察问题：OBSERVATION TOO LONG，说明 max_obs_length=256 偏小，后续正式实验建议调大。
最终状态：未完成 20 step。训练容器在 epoch 0 step 2 附近退出。
直接错误：Ray GCS 断连，日志出现 Failed to connect to GCS within 60 seconds / GCS may have been killed。
清理状态：训练容器已退出，retriever 容器已手动停止，8 张 A100 显存已回到 0 MiB。
后续建议：下一次先跑更保守的 2-3 step，增大 max_obs_length，降低生成长度/turn 数，并显式关闭或绕开 Ray metrics exporter 噪声；确认稳定后再回到 20-50 step。
```

2026-06-26 第一阶段 3-step 保守复测：

```text
目标：验证修复 W&B tracking 后，Qwen2.5-3B + PPO + Flat GPU retriever 能否稳定跑过多个 step。
实验名：phase1-flat-ppo-qwen2.5-3b-3step-1turn-stable-20260626_073454
W&B run：https://wandb.ai/xuhaoxuan-harbin-institute-of-technology/search-r1/runs/m34sg39i
训练日志：/mnt/xu/xu_exp/Search-R1/logs/phase1-flat-ppo-qwen2.5-3b-3step-1turn-stable-20260626_073454.outer.log
Ray 日志：/mnt/xu/xu_exp/Search-R1/logs/ray_phase1-flat-ppo-qwen2.5-3b-3step-1turn-stable-20260626_073454
retriever 日志：/mnt/xu/xu_exp/Search-R1/logs/retriever_flat_phase1_stable_20260626_072810.log
配置：TOTAL_STEPS=3，MAX_TURNS=1，TRAIN_BATCH_SIZE=8，MAX_RESPONSE_LENGTH=64，MAX_OBS_LENGTH=512。
结果：完成 step 1，W&B online run 创建成功，step 1 的 actor/critic/env 指标已进入 stdout。
W&B 实测：手动 wandb sync 后，run m34sg39i 从 running 变为 finished，但 history_keys_count 仍为 3，training_keys_count 仍为 0，Charts 仍不会自动出现训练曲线。
判断：不占 GPU 的 tracking-sanitize-check 可以写入 W&B history，但真实训练异常退出时训练指标仍只进 stdout；后续要继续查 veRL 训练主路径的 W&B log/flush 调用。
失败位置：epoch 0 step 2，critic update 附近。
直接错误：Worker PID 9178 无法连接 Ray GCS，随后 ActorDiedError。
Ray 现场：gcs_server.out 记录 node health check Deadline Exceeded、worker SYSTEM_ERROR；raylet.out 记录 worker connection error code 2 / End of file。
系统状态：无内核 OOM killer / NVIDIA Xid 记录；停止后 8 张 A100 显存为 0 MiB。
负载判断：失败时/失败后 A100 load average 极高，stop 后仍约 107 / 288 / 199，swap 几乎占满。这更像 Ray/GCS 在高 CPU/I/O/内存压力下失联，不像单纯 CUDA OOM。
清理状态：训练容器已停止，retriever 容器已停止，GPU 已释放；失败容器保留为 Exited(1) 便于后续 inspect。
下一步：不要在当前高负载状态继续重复同一配置；等负载下降后再跑 2-3 step，或进一步降低 Ray/训练压力后复测。
```

W&B Charts 只显示 System 的原因与修复：

```text
问题：失败 run 4ctwe3mp 的 stdout Logs 里有 step:1 指标，但 W&B historyKeys 只有 system 相关 key，没有 actor/critic/env。
确认方式：通过 W&B API 查询 run 4ctwe3mp，training_keys_count=0。
原因判断：训练在 Ray GCS 断连后异常退出，原 veRL W&B logging 没有把训练指标可靠写入 history；stdout 被 W&B 捕获，所以 Logs 里能看到，Charts 里看不到。
修复：已修改 /Users/xuhaoxuan/Desktop/deepseek_one_week/repos/Search-R1/verl/utils/tracking.py，并同步到 A100。
修复内容：对 wandb/mlflow backend 只上传数值标量，和 LocalLogger 的行为对齐。
验证 run：https://wandb.ai/xuhaoxuan-harbin-institute-of-technology/search-r1/runs/0buw20gt
验证结果：Charts 中能出现 actor/pg_loss、critic/vf_loss、env/finish_ratio。
注意：真实训练 run m34sg39i 在异常退出后手动 sync 仍没有 actor/critic/env history；说明 tracking-sanitize-check 的修复只能证明 W&B 后端可写，不能证明当前训练主路径在崩溃场景下一定 flush 成功。
```

2026-06-26 W&B history flush 修复：

```text
问题更新：10-step 成功训练 run b4zqlich 最初有 W&B 链接和 stdout Logs，但线上 Charts 缺少 actor/critic/env history，run state 显示 crashed。
根因判断：PPO 主入口正常 return 后没有显式 wandb.finish()，W&B 后台上传没有完整收尾。
代码修复：
- /Users/xuhaoxuan/Desktop/deepseek_one_week/repos/Search-R1/verl/utils/tracking.py 增加 Tracking.finish(exit_code=...)，wandb/mlflow/console 分别收尾。
- /Users/xuhaoxuan/Desktop/deepseek_one_week/repos/Search-R1/verl/trainer/main_ppo.py 和 main_ppo_format.py 用 try/finally 包住 init_workers()+fit()，正常 exit_code=0，异常 exit_code=1。
- /Users/xuhaoxuan/Desktop/deepseek_one_week/repos/Search-R1/scripts/xu/run_phase1_flat_ppo_safe_docker.sh 增加 WANDB_SYNC_AFTER=1，训练容器退出后用同一镜像自动 wandb sync 最新 run 目录作为兜底。
补救脚本：
- /Users/xuhaoxuan/Desktop/deepseek_one_week/repos/Search-R1/scripts/xu/wandb_backfill_scalar_history.py 可从 Search-R1 console log 的 step:N 标量文本补写 W&B history。
当前验证：
- smoke run fkn68lz6：线上 scan_history 可见 actor/pg_loss、critic/vf_loss、env/finish_ratio，state=finished。
- 10-step run b4zqlich：已 wandb sync 并 backfill step 9/10；线上 scan_history 共 10 行，steps=[1..10]，step 10 包含 val/test_score/nq=0、val/test_score/hotpotqa=0，state=finished。
```

当前 W&B 项目信息：

```text
项目 URL：https://wandb.ai/xuhaoxuan-harbin-institute-of-technology/search-r1
WANDB_ENTITY：xuhaoxuan-harbin-institute-of-technology
trainer.project_name / W&B project：search-r1
```

Search-R1 的 veRL tracking 只显式传 `project` 和 `name`：

```text
wandb.init(project=trainer.project_name, name=trainer.experiment_name, config=config)
```

因此 online 运行时需要同时设置：

```text
WANDB_ENTITY=xuhaoxuan-harbin-institute-of-technology
WANDB_MODE=online
WANDB_API_KEY=<从 /mnt/xu/xu_exp/Search-R1/.secrets/wandb.env 注入，不写入 README>
trainer.logger=['console','wandb']
trainer.project_name=search-r1
trainer.experiment_name=<本次实验名>
```

建议规则：

```text
smoke test：优先 trainer.logger=['console'] 或保持 WANDB_MODE=offline
正式实验：如果需要网页监控，Docker 启动时使用 --env-file /mnt/xu/xu_exp/Search-R1/.secrets/wandb.env
online 已通过最小测试，但大训练仍建议保留 console 日志作为兜底
校园网不稳定时：先离线记录，训练结束后再用 wandb sync 上传
```

A100 已启用 `nvidia-persistenced`。这会让 `nvidia-smi` 和 NVIDIA 容器运行时初始化保持快速；如果以后重启服务器后 GPU 查询又变慢，优先检查：

```bash
systemctl status nvidia-persistenced
```

## A100 显存占位脚本

已准备一个低功率显存占位脚本：

```text
本地脚本：/Users/xuhaoxuan/Desktop/deepseek_one_week/remote_scripts/hold_gpu_memory.py
A100 脚本：/mnt/xu/xu_exp/tools/hold_gpu_memory.py
```

原理：用 PyTorch 在每张 GPU 上分配 CUDA tensor，然后进程睡眠不退出。这样会占用显存，但不做训练/推理计算，所以通常不会明显拉高功率。退出进程后显存释放。

先 dry-run，不占显存：

```bash
ssh nuist-a100-via-windows 'source /mnt/xu/miniconda3/etc/profile.d/conda.sh && conda activate EnerGen && python /mnt/xu/xu_exp/tools/hold_gpu_memory.py --devices all --gib-per-gpu 1 --dry-run'
```

占 8 张 A100，每张尽量保留 4 GiB 空闲：

```bash
ssh -tt nuist-a100-via-windows 'source /mnt/xu/miniconda3/etc/profile.d/conda.sh && conda activate EnerGen && python /mnt/xu/xu_exp/tools/hold_gpu_memory.py --devices all --leave-free-gib 4 --heartbeat-seconds 300'
```

指定每张卡占 36 GiB：

```bash
ssh -tt nuist-a100-via-windows 'source /mnt/xu/miniconda3/etc/profile.d/conda.sh && conda activate EnerGen && python /mnt/xu/xu_exp/tools/hold_gpu_memory.py --devices all --gib-per-gpu 36 --heartbeat-seconds 300'
```

建议放在远端 `tmux` 里跑，避免本地断线导致进程退出：

```bash
ssh nuist-a100-via-windows
tmux new -s hold_gpu_mem
source /mnt/xu/miniconda3/etc/profile.d/conda.sh
conda activate EnerGen
python /mnt/xu/xu_exp/tools/hold_gpu_memory.py --devices all --leave-free-gib 4 --heartbeat-seconds 300
```

释放显存：

```text
在脚本窗口按 Ctrl+C
```

或：

```bash
tmux kill-session -t hold_gpu_mem
```

不要在共享机器上长期占卡，除非已经确认这 8 张卡可以由自己使用。

当前显存占位状态记录：

```text
nuist_remote_bridge/a100_gpu_hold_status.md
```

## 实验输出管理

远程实验输出默认放在项目目录下：

```text
/mnt/xu/xu_exp/项目名/outputs
/mnt/xu/xu_exp/项目名/checkpoints
/mnt/xu/xu_exp/项目名/logs
```

长期重要结果可以整理到：

```text
/mnt/xu/xu_exp/项目名/results
```

Mac 本地只同步必要的小结果，默认放：

```text
/Users/xuhaoxuan/Desktop/deepseek_one_week/outputs/项目名
```

不要把大 checkpoint、完整数据集、大日志直接拉回 Mac，除非用户明确要求。

## 文件传输规则

Mac 到 Windows 的中转目录：

```text
Mac:     /Users/xuhaoxuan/campus-transfer
Windows: D:\campus-transfer
```

Mac 传文件到 Windows：

```bash
scp -O /path/to/file windows-bridge:'D:/campus-transfer/'
```

Mac 直接经过 Windows 传到 A100：

```bash
scp -J windows-bridge /path/to/file xu@10.255.251.134:/target/path/
```

代码目录同步到 A100 时，优先用 `rsync`，并排除缓存、输出和 checkpoints。

3090/A20 到 A100 的大文件传输，优先让 A100 直接拉，不走 Mac 中转。

当前已配置的固定链路：

```text
3090 入口：XU_a20_cpolar / 7.tcp.vip.cpolar.cn:10966
3090 用户：xu
A100 用户：xu
A100 专用私钥：~/.ssh/id_ed25519_a100_to_3090_searchr1
3090 授权方式：A100 专用公钥已加入 3090 的 ~/.ssh/authorized_keys
```

适用场景：

```text
大 Docker tar
大模型权重
大数据压缩包
需要断点续传的文件
```

如果单个大文件通过 cpolar/单 TCP 流太慢，优先在 3090/A20 上处理后再传：

```text
1. 对模型、数据、索引目录先打 tar.zst，减少传输体积。
2. 如果仍然慢，再按固定大小切片，用多个 rsync 会话并行传。
3. 每个包或切片都保留 sha256，A100 收完后先校验再解压/合并。
4. 并行切片只用于传输瓶颈明确在 cpolar 单流时；如果瓶颈是总带宽，并行不会明显提升。
```

推荐方式是在 A100 上开 `tmux`，用 `rsync --append-verify` 从 3090 拉取：

```bash
tmux new-session -d -s rsync_searchr1_docker \
  "mkdir -p /mnt/xu/docker_imports; \
   rsync -avP --append-verify --info=progress2 \
     -e 'ssh -i ~/.ssh/id_ed25519_a100_to_3090_searchr1 -p 10966 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o StrictHostKeyChecking=accept-new' \
     xu@7.tcp.vip.cpolar.cn:/mnt/data/xu/projects/Search-R1-docker/searchr1_cu121_vllm063_flashattn.tar \
     /mnt/xu/docker_imports/ \
     2>&1 | tee /mnt/xu/docker_imports/searchr1_rsync_20260625.log"
```

检查进度：

```bash
ssh nuist-a100-via-windows 'tmux capture-pane -pt rsync_searchr1_docker -S -40'
ssh nuist-a100-via-windows 'tail -n 20 /mnt/xu/docker_imports/searchr1_rsync_20260625.log'
ssh nuist-a100-via-windows 'ls -lh /mnt/xu/docker_imports/searchr1_cu121_vllm063_flashattn.tar'
```

当前 Search-R1 Docker 镜像传输记录见：

```text
nuist_remote_bridge/search_r1_sync_log.md
```

## 南信大桥接状态

详细桥接配置见：

```text
nuist_remote_bridge/README.md
```

核心可用别名：

```bash
ssh windows-bridge
ssh nuist-a100-via-windows
ssh nuist-4090-via-windows
ssh XU_a20_cpolar
```

当前稳定链路：

```text
Mac/Codex
  -> 网线 IPv6 link-local
  -> Windows/ENAgent
  -> 南信大 A100
```

ENAgent 开启时，Windows 的 IPv4 SSH 可能超时；`windows-bridge` 已配置为走 IPv6 link-local。

如果网线被拔掉，一般重新插上即可。若 `windows-bridge` 失效，优先检查：

```text
1. Mac 直连网卡是否仍是 172.31.254.1
2. Windows 以太网是否仍是 172.31.254.2
3. Mac 的网卡接口名是否仍是 en7
4. Windows 的 IPv6 link-local 地址是否变化
```

## 远程盘点记录

A100 账号、磁盘、项目、数据和环境盘点见：

```text
nuist_remote_bridge/a100_account_inventory_20260624.md
```

当前重要结论：

```text
/mnt 总盘 109T，已用约 95T，使用率约 92%
/mnt/xu 总占用约 1.6T
/mnt/xu/xu_data 约 1.3T
/mnt/xu/xu_exp 约 276G
/mnt/xu/xu_exp/EnerGen/checkpoints 约 273G
```

`EnerGen` 工作区有大量改动和未跟踪文件。不要随便 `git reset`、删除 checkpoints 或清理结果目录，除非用户明确确认。

## 安全与清理规则

默认不做这些事：

```text
不删除远端数据
不清空 checkpoints
不重置 git 工作区
不覆盖用户已有改动
不把密码/私钥写进 README
不把大文件推到 GitHub
```

凭据只放在本地 ignored 文件：

```text
nuist_remote_bridge/secrets/credentials.local.md
```

如果需要清理空间，优先先列出候选项和大小，再让用户确认。

优先检查对象：

```text
/home/xu/.cache/pip/http-v2
/mnt/xu/xu_exp/项目名/checkpoints
/mnt/xu/xu_exp/项目名/outputs
```

## Codex 默认工作方式

用户说“同步到 A100”“在服务器跑”“检查远端”时，默认理解为：

```text
目标服务器：nuist-a100-via-windows
目标账号：xu
代码目录：/mnt/xu/xu_exp/项目名
数据目录：/mnt/xu/xu_data
环境目录：/mnt/xu/miniconda3
```

如果用户没有特别指定，Codex 应：

```text
1. 每次重新唤醒或开始新一轮远端工作时，先检查链路：
   - ssh windows-bridge
   - ssh nuist-a100-via-windows
   - 必要时顺手确认远端 Docker / GPU 状态
2. 先检查本地项目和远端路径
3. 避免覆盖远端未确认的重要输出
4. 同步代码时排除 .git、缓存、outputs、checkpoints
5. 跑实验前显式激活 conda 环境
6. 跑实验后检查日志、进程、GPU 和输出文件
7. 把关键状态更新到相关 README 或记录文件
```
