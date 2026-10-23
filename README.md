# SmolVLA 机器人操作实验室

本项目用于学习和展示一条完整的 VLA 实验流程：**检查机器人轨迹 → 微调策略 → 在仿真环境中闭环执行 → 分析效果、资源开销与失败原因**。

实验采用 SmolVLA、LeRobot 和 LIBERO。两张 RTX 4090 分别运行默认动作专家微调与 LoRA 微调，属于两个独立的单卡实验，**不是双卡分布式训练**。

**交付时状态：**模型与数据准备、整轨数据检查、EGL 渲染、两组20步短训练、checkpoint 重载及40步仿真回合均已通过。两组5000步正式实验已启动，尚无正式闭环成功率。本文的配置是实验设计，实际结果以 `status.py`、日志和评测文件为准。

先看 `INTERVIEW_GUIDE.md`，按4天路线理解代码；当前项目的个人工作是数据适配、统计与划分、实验组织、证据记录和结果分析。

## 实验目标与范围

| 项目 | 当前设置 |
|---|---|
| 服务器项目目录 | `/root/autodl-tmp/smolvla_lab` |
| LeRobot 固定提交 | `f9e9d739ea7055b282ea1b165e8785412a674693` |
| Python / 主要依赖 | 项目虚拟环境；Torch 2.8.0+cu128；Transformers 5.5.4；完整版本见环境报告 |
| 基础策略 | `lerobot/smolvla_base`，提交 `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6` |
| 数据 | `lerobot/libero`，提交 `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` |
| 任务 | LIBERO-Spatial 的任务编号 `0,1,2,3,4` |
| 示范划分 | 每任务20条训练、5条验证；共100条训练、25条验证轨迹 |
| 初版训练 | 每组5000步，batch size 16，训练 seed 1000 |
| 闭环评测 | 每任务20回合，每方法共100回合，评测 seed 2000 |
| 动作配置 | `chunk_size=50`、`n_action_steps=10`、`num_steps=10` |

这是一个**单个训练种子的少样本初版实验**。它只能说明所选5个任务和当前设置下的表现，不是完整 LIBERO 结果，也不以复现论文最佳成绩为目标。后续应增加训练种子、评测覆盖和任务数量，再讨论结论的稳定性。

## 两组方法在比较什么

| 组别 | 物理 GPU | 训练方式 | 学习率 |
|---|---|---|---|
| `baseline` | GPU 0 | 默认动作专家微调；冻结视觉编码器，启用动作专家与状态投影训练；99,880,992个可训练参数 | `1e-4` |
| `lora` | GPU 1 | 官方 PEFT LoRA 配方，rank 32，alpha 32；1,485,312个可训练参数 | `1e-3` |

以上参数数目已在短训练中实测。两组使用相同的基础权重、数据划分、步数、batch size、训练 seed 和闭环评测设置。实际可训练参数名称与数量会保存到报告。

**这是一组实用训练配方比较，不是严格只改变 LoRA 一个变量的消融实验。** 学习率也随配方变化，因此不能把所有效果差异都归因于 LoRA。默认组也不应称为“整个 VLA 全量微调”。

LoRA 的默认目标包括动作专家中的注意力投影，以及状态/动作相关投影层；最终以参数报告为准。它可能降低训练资源需求，也可能损失任务效果，评价时应同时看成功率、参数量、显存和耗时。

## 首次准备

这台服务器已完成准备，无需重复下载。新服务器需先安装 `ffmpeg libegl1 libegl-mesa0 libgl1-mesa-dri libosmesa6 libglib2.0-0`，然后执行 `bash bootstrap.sh`。下列命令均在云服务器中执行。

```bash
cd /root/autodl-tmp/smolvla_lab
source env.sh

# 先写好 LIBERO 路径，避免首次导入时出现交互提示。
python scripts/setup_libero.py --configure-only

# 新环境可从官方固定版本取得仿真资源；当前服务器已经备好。
curl -L --fail --retry 3 \
  https://codeload.github.com/Lifelong-Robot-Learning/LIBERO/tar.gz/8f1084e3132a39270c3a13ebe37270a43ece2a01 \
  -o vendor/libero-master.tar.gz

# 下载基础策略及其依赖的视觉语言模型，并记录固定版本。
python scripts/prepare_models.py

# 下载数据，筛选轨迹，计算训练统计，并导出数据检查报告。
python scripts/prepare_data.py

# 验证无显示器的 EGL 渲染、环境 reset 和一次动作执行。
python scripts/setup_libero.py
```

每次重新登录后，都应进入项目目录并执行 `source env.sh`。它配置虚拟环境、项目内缓存、EGL 和本地日志模式。默认下载入口使用 Hugging Face 镜像；模型和数据来源及提交号仍会记录在报告中。

当前资源包最初从官方 master 下载，其585个资源文件已逐一按 Git blob 哈希验证，与上述固定提交完全一致，见 `reports/libero_assets_verification.json`。资源包与大型模型保存在服务器，未包含在本机的轻量交付副本中。

**数据格式适配：**固定版本的源数据缺少 episode 的 `tasks` 列。脚本通过帧级 `episode_index/task_index` 和任务表恢复该列，保留原始 parquet 备份与修复前后哈希。修改只作用于本地副本，见 `reports/episode_metadata_repair.json`。

数据筛选以任务的自然语言指令匹配 LIBERO 的任务定义，并保存轨迹编号。训练与验证按完整轨迹划分，不会随机拆分同一轨迹的相邻帧。

**归一化防泄漏：**原始数据统计先备份到 `reports/source_stats.json`；项目重新计算仅来自100条训练轨迹的状态/动作统计，并写入项目内的数据副本。25条验证轨迹不参与这些统计；图像统计保持原值，SmolVLA 的视觉归一化采用 `IDENTITY`。脚本不向 Hub 上传或修改远程数据。

下载阶段若中断，应先检查日志与磁盘空间，再重新执行相应准备命令。已有数据若固定提交号与新请求不一致，脚本会拒绝混用，需使用新的项目目录或明确采用原版本。

## 先做流程检查

```bash
python scripts/launch.py --phase smoke --steps 20 --batch-size 16 --smoke
python scripts/status.py
```

这会在两个 GPU 上各启动20步训练，随后各做任务0的一次短回合评测，回合长度限制为40步。其目的仅是验证**读数据、前向反向、保存模型和仿真执行**这条链路；短回合失败不等于正式策略失败，成功也不能作为模型效果结论。

启动器会将实验放到独立进程，SSH 断开后通常仍继续运行。关闭或释放云服务器会中止实验。

## 启动正式初版

确认流程检查没有报错，并且两张 GPU 没有仍在执行前一轮实验后，运行：

```bash
python scripts/launch.py --phase phase1 --steps 5000 --batch-size 16
```

每组独立进程依次完成以下流程，无需另行手工启动评测：

1. GPU 0 训练 `phase1_baseline`，GPU 1 训练 `phase1_lora`。
2. 保存训练过程指标、参数报告和 checkpoint。
3. 训练结束后，用最终 checkpoint 进行5任务闭环评测。
4. 保存逐回合与汇总评测结果，以及评测视频。

训练阶段保存周期为总步数的一半，并在结束时保存最终点；初版约在2500、5000步保存。训练还会定期计算有限数量的验证样本损失。**验证损失是离线指标，闭环成功率来自独立环境执行，两者不能混用。**

本次还已启动 `finish_phase.py`，等待两组结束后，在同一张空闲 GPU 上依次测量10/5采样步的动作生成时延，再生成 `reports/phase1/EXPERIMENT_REPORT.md`。它是服务器上的普通进程，不会向外发送通知。以后新开 phase 时可运行：

```bash
nohup python scripts/finish_phase.py --phase phase2 > logs/phase2_finalize.log 2>&1 < /dev/null &
```

## 查看进度与定位错误

```bash
python scripts/status.py
tail -n 80 logs/phase1_baseline.log
tail -n 80 logs/phase1_lora.log
python scripts/summarize.py --phase phase1
df -h /root/autodl-tmp/smolvla_lab
```

`status.py` 展示阶段、已完成训练步数、目标步数、训练时间、显存峰值和错误信息，并展示两张 GPU 的当前状态。阶段含义：

| 阶段 | 含义 |
|---|---|
| `training` | 训练进程已启动；仍需结合进度文件和日志判断是否持续推进 |
| `evaluating` | 训练结束，正在运行闭环评测 |
| `completed` | 该实验的训练和后续评测命令均已正常退出；任务效果仍需读结果 |
| `failed` | 某一步报错，检查对应日志末尾及状态文件中的错误 |

本机 `reports/` 是交付时快照；实时进度请在服务器查看。若没有任何实验记录，不能据此断言准备成功。下载、依赖或流程检查失败时，也应先修复后再启动正式实验。磁盘只有有限空间，checkpoint、优化器状态、模型缓存和视频都会占用空间，运行新实验前应检查剩余容量。

## 成果在哪里

以下路径均相对于服务器项目目录：

| 路径 | 内容与用途 |
|---|---|
| `reports/runtime.json`、`reports/environment.freeze.txt` | GPU、依赖版本和环境记录 |
| `reports/lerobot_commit.txt` | 实际使用的 LeRobot 提交 |
| `reports/models.json` | 模型来源、固定提交号、权重哈希及本地配置调整 |
| `configs/data_selection.json` | 精确任务、训练/验证轨迹编号、数据版本与统计范围 |
| `reports/data_audit.json` | 数据维度、时序检查、异常检查、训练分布与划分依据 |
| `reports/train_only_stats.json`、`reports/source_stats.json` | 训练集统计与原始统计备份 |
| `reports/*_checkpoint_audit.json` | 保存后的模型接口、归一化及反归一化统计核对 |
| `reports/sample_cameras.png` | 一条实际数据样本的两路相机预览 |
| `reports/libero_probe.json`、`reports/libero_assets.json` | EGL reset/step 检查、官方资源来源及哈希 |
| `reports/phase1/EXPERIMENT_REPORT.md` | 根据实际文件汇总的正式实验报告；未结束时显示待完成 |
| `reports/phase1_finalize.json`、`logs/phase1_finalize.log` | 自动测速与汇总流程状态及日志 |
| `logs/phase1_baseline.log`、`logs/phase1_lora.log` | 两组完整运行日志 |
| `reports/phase1_*_parameters.json` | 总参数量、可训练参数量及实际参数名称 |
| `reports/phase1_*_metrics.jsonl` | 训练指标、训练耗时和显存峰值的持续记录 |
| `reports/phase1_*_status.json`、`reports/phase1_*_progress.json` | 流程状态与最新训练进度 |
| `runs/phase1_*/checkpoints/last/pretrained_model/` | 最终模型或 LoRA adapter，以及对应配置和 processor |
| `runs/phase1_baseline_eval/`、`runs/phase1_lora_eval/` | `eval_info.json`、评测视频等闭环结果 |

LoRA checkpoint 仍依赖其基础模型。迁移实验时，要同时保留基础权重、adapter 配置、processor 和所用基础模型路径的可用性。

## 可选扩展：采样计算与耗时

正式评测完成后，可在空闲 GPU 上测量动作序列生成耗时：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_latency.py \
  --checkpoint runs/phase1_baseline/checkpoints/last/pretrained_model \
  --name phase1_baseline

CUDA_VISIBLE_DEVICES=1 python scripts/benchmark_latency.py \
  --checkpoint runs/phase1_lora/checkpoints/last/pretrained_model \
  --name phase1_lora
```

脚本比较 `num_steps=10` 和 `5`，先预热并同步 GPU，再记录均值、中位数、P95与原始时间。输出为独立的 `reports/<name>_latency_<timestamp>_<nonce>.json`。`smoke_codecheck_*` 只用于检查测速脚本，运行时有其他训练负载，不用于性能比较。**这里仅测模型生成动作序列的耗时，不包含数据预处理、环境渲染和完整任务耗时。**

要判断减少采样步数是否值得，还应额外运行相同初始状态下的闭环成功率对照；仅有测速不能证明任务执行得到改善。`num_steps` 是 Flow Matching 的积分采样步数，`n_action_steps` 是重新观察环境前执行多少个动作，二者不是同一参数，做对照时一次只改变一个。

## 重跑、延长与恢复

启动器拒绝覆盖已有实验名。若需要从基础模型重新做更长训练，应使用新 phase，例如：

```bash
python scripts/launch.py --phase phase2 --steps 10000 --batch-size 16
```

**这会从基础模型重新开始，不会自动接着 phase1 的5000步训练。** 真正续训需使用官方 LeRobot 的 `--resume=true`，并将 `--config_path` 指向该 checkpoint 的 `pretrained_model/train_config.json`；保留其 `training_state`、优化器、scheduler和随机状态，并核对恢复后的总步数及学习率计划。不要通过删除已有实验记录来绕过覆盖保护。

若只是资源下载或渲染导致评测中断，训练 checkpoint 已存在，可用 `launch.py --phase <原phase> --eval-only` 重试闭环评测；它不会重训。smoke 重试同时加 `--smoke`。必须先确认原评测进程已经退出。

初版只有一个训练 seed。增加训练种子或更改数据划分时，还应修改相应配置和实验命名，记录变化；不能只改名称便宣称结果来自多个随机种子。

## 一页面试理解检查清单

能够结合实际代码、日志和结果回答以下问题，才算理解这个项目：

- **任务与贡献：**固定的5个任务是什么？哪些工作来自 LeRobot，哪些数据检查、配置、记录和分析由本项目完成？
- **数据：**8维状态和7维动作分别是什么？两路相机与动作如何对齐？为什么按整条轨迹划分？训练统计为什么不能包含验证轨迹？
- **模型：**图像、指令、机器人状态如何进入 VLA？动作专家生成什么？Flow Matching 学习的目标与普通动作回归有什么区别？
- **动作：**`chunk_size=50`、`n_action_steps=10`、`num_steps=10`各表示什么？更久才重新观察有什么收益和风险？
- **训练：**默认组实际冻结了什么？LoRA 的目标层、rank、alpha和参数量是什么？本次为什么不能将所有组间差异归因于 LoRA？
- **评测：**离线验证损失和闭环成功率衡量什么？如何固定任务、初始状态和seed？为什么5任务、单训练种子的结果不能等同完整benchmark？
- **资源：**峰值显存、训练时长和模型推理时长如何测量？GPU异步执行为什么影响计时？两卡独立实验与DDP有什么区别？
- **失败：**能否展示一个实际失败回合，提出可检验的原因，并设计下一步对照？有没有把模型问题与接口、归一化或执行配置问题区分开？
- **结论边界：**低分或未提升时怎样如实解释？更高成功率是否付出了更多计算成本？哪些结论尚需更多seed或任务验证？

整理简历时填写实际跑出的数据。可以将项目表述为“基于开源 VLA 的适配、对照与闭环分析”，不要写成从零预训练、真机部署、分布式训练或已经达到论文最佳结果。

## 官方资料

- [SmolVLA 模型与训练](https://huggingface.co/docs/lerobot/smolvla)
- [SmolVLA LoRA/PEFT 配方](https://huggingface.co/docs/lerobot/peft_training)
- [LIBERO 数据与评测](https://huggingface.co/docs/lerobot/libero)
- [LeRobot 算力参考](https://huggingface.co/docs/lerobot/hardware_guide)
- [本项目固定的 LeRobot 源码](https://github.com/huggingface/lerobot/tree/f9e9d739ea7055b282ea1b165e8785412a674693)

官方文档会更新。涉及字段、行为和默认值时，以本项目固定提交及实际保存的配置为准。
