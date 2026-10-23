# SmolVLA 项目精读与面试指南

适用背景：厦门大学研一，有 CNN/ResNet 和智能车竞赛经历，准备在 1–2 周内申请具身智能/VLA 算法实习。

这份指南围绕本项目实际脚本和固定版本的 LeRobot 源码展开。目标是让你能解释数据、模型、训练、仿真执行和实验边界，并能从日志找到证据。不要把“背过答案”当成“做过实验”。

## 先确认当前事实

- 本项目使用开源 SmolVLA 基础权重、LeRobot 的策略/训练器/PEFT 实现，以及 LIBERO 仿真任务。
- 初版选择 LIBERO-Spatial 的任务 `0,1,2,3,4`。每任务 20 条训练轨迹、5 条验证轨迹，共 100 条训练、25 条验证。
- 两组正式初版均已完成 5000 次更新，batch size 16、训练 seed 1000。每组完成五项 Spatial 任务、每任务 20 回合的闭环评测：默认动作专家 **76/100** 成功，LoRA **63/100** 成功。
- 两张 RTX 4090 分别运行默认动作专家微调与 LoRA。它们是两个独立单卡实验，没有联合更新同一个模型。
- 正式参数报告记录默认组可训练参数 **99,880,992**；LoRA 组可训练参数 **1,485,312**。这与流程检查一致，但不能把参数减少比例直接换算成显存或速度收益。
- 默认组和 LoRA 使用不同学习率，属于实用配方比较；不能把观察到的 13 个百分点成功率差异全部归因于 LoRA。
- 训练使用 **BF16 混合精度**。正式闭环评测与主推理时延对照使用 **FP16 autocast**；早期 BF16 测速如有保留，应单独标注，不与 FP16 混合汇总。
- 相同 baseline 最终 checkpoint 的 `num_steps=10→5` 采样对照已完成，唯一干预为积分采样步数，`n_action_steps=10` 等设置保持固定。FP16、GPU 0 下，两组均成功 **76/100**；动作块生成中位时延 **349.9760→201.2357 ms，下降42.50%**。这里的统计量是两种测试顺序各自中位数的中位数，不是完整任务耗时。
- 采样对照满足预设的时延下降至少20%、总成功率最多下降5个百分点、任一任务最多下降10个百分点的工程规则。其中任务0恰好下降10个百分点，任务4上升10个百分点；总成功数相同不代表逐任务或逐回合结果完全相同。不能称统计上无损或非劣效。
- 三组闭环评测共 **300 条视频**已保留并通过正式审计；原视频与展示视频分别记录 SHA256，初值、seed、成功标签和展示来源的映射已经核查。按20 Hz仿真控制时间播放的视频不包含模型推理等待。
- 当前没有真机部署、Sim2Real、强化学习后训练、论文创新或分布式训练成果。

| 正式闭环任务 | 默认动作专家，10步采样 | LoRA，10步采样 | 同基线权重，5步采样 |
|---|---:|---:|---:|
| Spatial 0 | 17 / 20 | 10 / 20 | 15 / 20（75%） |
| Spatial 1 | 12 / 20 | 10 / 20 | 12 / 20（60%） |
| Spatial 2 | 18 / 20 | 19 / 20 | 18 / 20（90%） |
| Spatial 3 | 18 / 20 | 17 / 20 | 18 / 20（90%） |
| Spatial 4 | 11 / 20 | 7 / 20 | 13 / 20（65%） |
| 合计 | **76 / 100** | **63 / 100** | **76 / 100** |

| FP16动作块生成微基准，GPU 0 | 中位时延（ms） | 对应闭环结果 |
|---|---:|---|
| 默认动作专家，10步采样 | 349.9760 | 76 / 100 |
| 默认动作专家，5步采样 | 201.2357 | 76 / 100 |
| LoRA，10步采样 | 407.1434 | 63 / 100 |
| LoRA，5步采样 | 233.6729 | 未做该配置的闭环评测，仅测速 |

时延在固定单个验证观测、batch size 1 下测量，包含预热、GPU同步和两种设置顺序的复测。表中时间不含预处理、仿真、渲染和动作后处理，不能用来描述完整机器人响应或所有输入的时延分布。

总体成功率的官方 95% Wilson 区间分别约为 66.8%–83.3% 和 53.2%–71.8%。区间来自已测回合，不能代替跨训练种子和任务的重复实验，也不据此单独判断两配方的统计显著差异。

已完成事实来源为 `reports/phase1_baseline_parameters.json`、`reports/phase1_lora_parameters.json`，以及 `runs/phase1_baseline_eval/eval_info.json`、`runs/phase1_lora_eval/eval_info.json`、`runs/sampling5_baseline_eval/eval_info.json`。采样对照与时延原始文件见 `reports/sampling5_baseline.json`、`reports/controlled_*_latency_*.json`；闭环与视频审计见 `reports/evaluation_audit.json`、`reports/media_manifest.json`。面试前重新核对原始记录，不把本文件当实时状态。

### 阅读路径怎么对应

本文件中的 `scripts/`、`configs/`、`reports/` 都相对于项目根目录。云服务器项目根目录为 `/root/autodl-tmp/smolvla_lab`。`vendor/lerobot/` 指服务器上的官方源码目录，固定提交为 `f9e9d739ea7055b282ea1b165e8785412a674693`。

| 要理解的内容 | 项目脚本或记录 | 官方源码入口 |
|---|---|---|
| 任务筛选、轨迹划分、训练集统计 | `scripts/prepare_data.py`；`configs/data_selection.json`；`reports/data_audit.json` | `vendor/lerobot/src/lerobot/datasets/factory.py` 的 `make_train_eval_datasets` |
| 两路相机、机器人状态 | `reports/sample_cameras.png`；`reports/data_audit.json` | `vendor/lerobot/src/lerobot/envs/libero.py`；`vendor/lerobot/src/lerobot/processor/env_processor.py` |
| 图像/指令/状态预处理 | `scripts/run_experiment.py` 的输入输出配置 | `vendor/lerobot/src/lerobot/policies/smolvla/processor_smolvla.py` |
| 模型与动作专家 | 两组保存的策略配置、参数报告 | `vendor/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py` |
| Flow Matching 目标与采样 | `chunk_size`、`num_steps` 配置 | 同上，以及 `vendor/lerobot/src/lerobot/policies/common/flow_matching.py` |
| 微调、LoRA 与训练证据 | `scripts/run_experiment.py`；`scripts/train_entry.py`；`reports/*_parameters.json` | SmolVLA 的 `_get_default_peft_targets`；`vendor/lerobot/src/lerobot/scripts/lerobot_train.py` |
| 并行运行和进度 | `scripts/launch.py`；`scripts/status.py` | 本项目的进程编排，未实现 DDP |
| 闭环评测 | `scripts/run_experiment.py` 的评测命令；`runs/phase1_*_eval/eval_info.json` | `vendor/lerobot/src/lerobot/scripts/lerobot_eval.py` |
| 模型耗时测量 | `scripts/benchmark_latency.py` | SmolVLA 的 `predict_action_chunk`、`sample_actions` |

## 四天精读路线

每天建议留出约 3–5 小时。现在可以对照已完成的正式训练、三组闭环结果、五步采样与主时延对照精读，不需要重新启动或覆盖实验。

### 第一天：把一条数据和完整流程讲清楚

**先读：** `README.md` → `configs/data_selection.json` → `reports/data_audit.json` → `scripts/prepare_data.py`。

**重点看：** `normalize_instruction`、`repair_episode_tasks`、`load_numeric_rows`、`vector_stats`，以及 `main` 中按任务划分轨迹的部分。

**当天完成：**

1. 用自己的话写出“语言任务 → 轨迹编号 → 图像/状态/动作 → 训练样本”的关系。
2. 打开 `reports/sample_cameras.png`，说明主视角和腕部视角分别能提供什么信息。
3. 从 manifest 找出一个任务的训练和验证轨迹编号，确认两者不重叠。
4. 解释为什么状态是 8 维，动作是 7 维；不要把状态误认为机械臂的 8 个关节角。
5. 对照 `make_train_eval_datasets`，解释项目为何按每任务“20 条训练在前、5 条验证在后”传入 episode 列表，并设置 `eval_split=0.2`。

**验收：** 不看文档，用 90 秒解释本项目的输入、输出、数据规模与划分。能指出 `prepare_data.py` 验证了帧号和时间戳连续性，但没有因此证明相机标定正确或真实物理同步精确。

**与你旧项目的连接：** CNN 项目中的图像张量、归一化和训练/验证划分经验可以直接迁移。智能车中的传感器、状态反馈、控制命令和日志经验可帮助理解策略闭环，但不能替代机械臂真机经验。

### 第二天：把模型、Flow Matching 和动作分块讲清楚

**先读：** `processor_smolvla.py` → `modeling_smolvla.py` 中的 `prepare_state`、`prepare_action`、`embed_prefix`、`embed_suffix`、`forward`、`sample_actions`、`select_action`。

**再读：** `policies/common/flow_matching.py` 的 `make_flow_matching_inputs` 和 `euler_integrate`。

**当天完成：**

1. 画出“图像编码、语言嵌入、状态投影 → 条件特征；带噪动作＋时间 → 动作专家 → 速度场”的数据流。
2. 标出状态 `[B,8]`、动作监督 `[B,50,7]` 与模型内部 padding 的区别。
3. 手写本文第 4 问的训练公式和推理更新式，说明时间方向。
4. 分别解释 `chunk_size=50`、`n_action_steps=10`、`num_steps=10`。
5. 沿 `select_action` 的队列逻辑，说明什么时候重新生成动作。

**验收：** 能解释“预测 50 步动作、先执行 10 步再重新观察”，并能说明 10 次 Flow Matching 采样不是执行 10 个机器人动作。知道本项目使用官方实现；没有自行提出或重写模型架构。

### 第三天：把训练、冻结和 LoRA 对照讲清楚

**先读：** `scripts/run_experiment.py` → `scripts/train_entry.py` → 两个 `reports/phase1_*_parameters.json`。

**当天完成：**

1. 列出两组相同的设置和不同的设置，尤其是学习率及衰减末值。
2. 从 `trainable_names` 找到默认组动作专家、状态/动作/时间投影参数，以及 LoRA 的 A/B 矩阵。
3. 手写 `W' = W + (alpha/r)BA`，解释 rank、alpha、可训练参数量和基础权重的关系。
4. 读 `train_entry.py`，说明它调用官方训练器，只增加参数、进度、显存等证据记录；它没有实现新的训练算法。
5. 看 `launch.py`，说明为什么两个独立进程并不等于双卡 DDP。

**验收：** 能准确说出“默认动作专家微调对比基于官方PEFT的LoRA配置”，并主动说明学习率不同，因此这不是严格只改变一个变量的 LoRA 消融。不能称默认组为“整个 VLA 全量微调”。

### 第四天：把评测、耗时、失败与贡献讲清楚

**先读：** `run_experiment.py` 的评测命令 → `scripts/benchmark_latency.py` → `scripts/summarize.py`。

**当天完成：**

1. 写清离线验证损失和闭环成功率各自怎样产生。
2. 核对闭环任务、回合数、seed、初始化、hard reset 和 batch 设置；两组比较时保持一致。
3. 从测速脚本指出 warmup、GPU 同步、显式精度选择、原始时间保存和测量范围；说明训练 BF16 与正式闭环/主时延 FP16 的区别。
4. 对照 `reports/FAILURE_ANALYSIS.md`，选择一个已记录的 baseline 失败回合写“现象—假设—证据—下一步实验”。三个案例不是全部失败的分类；把可见现象与原因猜测分开。
5. 准备 3 分钟项目介绍：任务与范围 30 秒，数据与训练 60 秒，评测设计 45 秒，个人工作与限制 45 秒。

**验收：** 能解释默认组 76/100、LoRA 63/100 的实测结果及配方比较边界；能说明同基线权重10→5步采样后总成功率均为76/100、中位模型时延下降42.50%，但任务0下降10个百分点、任务4上升10个百分点，因此不能将总成功数相同表述为统计上无损或所有任务表现不变。

## 十二个核心面试问题

### 1. 这个项目解决什么问题？哪些是你的工作？

**答题要点：**

- 在固定的少量机器人示范上，对比 SmolVLA 默认动作专家微调和基于官方PEFT的LoRA配置，并设计一致的仿真闭环评测。
- 开源部分是基础权重、VLM/动作专家、Flow Matching、PEFT、官方训练器和 LIBERO 环境。
- 本项目增加了任务与轨迹筛选、训练统计防泄漏、元数据兼容修复、实验配置和独立进程编排、参数/进度/显存记录，以及限定范围的耗时测量。
- 在同一最终基线 checkpoint 上完成10→5步的采样参数对照，并保留三组共300条闭环视频、初值与结果标签和来源哈希映射。这是对开源采样器的受控工程实验，不是新采样算法。
- 面试中的“我负责”只能包含自己实际完成或已理解、验证的工作。AI 辅助生成的脚本，应能解释和修改，不能据此宣称原创算法。

**证据：** `README.md`；`scripts/prepare_data.py`；`scripts/train_entry.py`；`scripts/benchmark_latency.py`。

**继续追问：** 如果去掉你写的脚本，官方代码能做什么？你的工作减少了哪些数据、复现或比较风险？

### 2. 输入和输出是什么？为什么状态 8 维、动作 7 维？

**答题要点：**

- 两路 RGB 图像分别是主视角 `observation.images.image` 和腕部视角 `observation.images.image2`；检查样本的两路图像都是 `[3,256,256]`。
- 语言输入是当前任务的自然语言指令。
- 8 维状态由末端位置 3 维、轴角姿态 3 维、夹爪两指位置 2 维组成。环境原始姿态使用四元数，processor 将其转换为轴角。
- 7 维动作是末端相对位姿控制的 6 维命令和 1 维夹爪命令，不能当作“7 个关节的目标角”。具体量纲、缩放与控制语义需要同时看环境控制器和数据定义，不能只凭维数推断。
- 模型一次生成动作序列。本配置对外动作块形状为 `[B,50,7]`。内部可能 pad 到更大的状态/动作宽度，这不改变环境实际接口。

**证据：** `reports/data_audit.json` 的 `sample`；`vendor/lerobot/src/lerobot/processor/env_processor.py` 的状态拼接；`vendor/lerobot/src/lerobot/envs/libero.py` 的相机映射和动作空间；SmolVLA 的 `prepare_state`、`prepare_action`。

**继续追问：** 主相机与腕部相机作用有什么不同？如果相机键顺序错误、图像旋转错误，训练 loss 一定会提示吗？

### 3. 观测如何进入模型？推理时需要真实动作吗？

**答题要点：**

- processor 处理观测键、batch 维度、语言末尾换行、tokenization、设备和归一化；动作后处理恢复环境使用的尺度。
- 图像经视觉编码器得到视觉特征，语言经 token embedding，状态经 `state_proj`。它们提供动作生成的条件。
- 动作专家接收带噪动作和时间嵌入，预测速度场。推理从随机噪声开始，不读取未来真实动作。
- 训练时真实动作作为监督；推理输入只能含当前观测与任务。`benchmark_latency.py` 显式排除了真实 `action` 和轨迹索引等辅助字段。
- 本项目启动训练时将 `policy.input_features/output_features` 置空，按目标数据重建接口，避免沿用基础模型原机器人的维度/键。统计和 processor 也必须与训练配置配套保存。

**证据：** `scripts/run_experiment.py`；`scripts/benchmark_latency.py` 中 `observation` 的构造；`processor_smolvla.py`；`modeling_smolvla.py` 的 `embed_prefix`、`embed_suffix`。

**继续追问：** ResNet 特征和 ViT/视觉 Token 有何区别？如果状态维度正确、归一化统计错误，可能出现什么现象？

### 4. Flow Matching 学什么？请写出本代码使用的公式。

令 `a` 为真实动作块，`epsilon ~ N(0,I)` 为同形状噪声，`c` 为图像/语言/状态条件。本实现采用 `NOISE_AT_ONE`：

```text
x_t = (1 - t) * a + t * epsilon
u_t = epsilon - a
L = E[ ||v_theta(x_t, t, c) - u_t||^2 ]
```

**答题要点：**

- `t=0` 对应真实动作，`t=1` 对应噪声。模型回归插值路径上的速度，不是简单回归一个固定动作，也不是把目标误写成噪声本身。
- 训练随机采样噪声与时间，每个样本可以使用不同的时间。当前时间采样由 `sample_time` 实现，使用缩放后的 Beta 分布；不应把代码说成均匀采样。
- 网络输出与速度目标计算逐元素 MSE。策略外层会裁掉无效动作维度，并处理超出轨迹边界的 padding。
- 训练通常不需要完成全部 10 次采样；`num_steps=10` 主要控制推理积分次数。

**证据：** `modeling_smolvla.py` 的策略 `forward`、模型 `forward`、`sample_time`；`policies/common/flow_matching.py` 的 `make_flow_matching_inputs`。

**继续追问：** 有些论文把噪声放在 `t=0`，为什么公式符号会相反？关键是训练路径、速度目标和推理方向必须一致。

### 5. 推理怎样从噪声得到动作？三个动作参数有什么区别？

**答题要点：**

- 推理从 `x_1 = epsilon` 开始，用当前观测计算条件，再从 `t=1` 积分到 `t=0`。均匀 Euler 步长是 `dt=-1/N`，更新为 `x_next = x + dt*v_theta(x,t,c)`。
- 图像与语言前缀的 KV Cache 可在同一动作块的积分过程中复用，减少重复计算；不能把它说成整个任务中永久不更新图像。
- `chunk_size=50`：一次预测的动作块长度。
- `n_action_steps=10`：同步执行时先取动作块前 10 步，队列耗尽后基于新观测重新预测。
- `num_steps=10`：生成该动作块所用的积分采样次数。它不是控制频率，也不是机器人执行步数。
- 执行动作更多可以摊薄预测开销，但减少反馈更新机会；具体是否更好要闭环验证。本项目没有因此实现异步推理或 RTC。

**证据：** `scripts/run_experiment.py` 的策略配置；`modeling_smolvla.py` 的 `sample_actions`、`select_action`；`flow_matching.py` 的 `euler_integrate`。

**继续追问：** 推理 3Hz、底层控制 50Hz 时怎么衔接？可以讨论动作队列、分块、插值、异步和安全约束，但必须区分设想与本项目已实现的同步执行。

### 6. 默认组冻结什么？LoRA 改了哪里？参数量有证据吗？

**答题要点：**

- 默认组设置 `freeze_vision_encoder=true`、`train_expert_only=true`、`train_state_proj=true`。参数报告显示动作专家层、状态投影、动作输入/输出与时间投影参与训练；不是整个 VLA 全量训练。
- LoRA 使用官方默认 target：动作专家注意力中的 `q_proj/v_proj`，以及 `state_proj`、`action_in_proj`、`action_out_proj`、`action_time_mlp_in/out`。不是笼统地“给视觉语言大模型所有层加 LoRA”。
- LoRA 把权重更新写成 `W' = W + (alpha/r)BA`，本配置 `r=32`、`alpha=32`，基础权重冻结，训练低秩 A/B 参数。
- 正式参数报告实测：默认组总参数 450,046,176、可训练 99,880,992；LoRA 组含 adapter 总参数 451,531,488、可训练 1,485,312。LoRA 总参数稍增加，而可训练部分大幅减少，二者不矛盾。
- 可训练参数少，通常减少梯度和优化器状态开销，但整体显存仍包含基础权重、激活和缓存。不能仅凭参数比率宣称等比例省显存或加速。
- 本次固定单观测、FP16、GPU 0 的实测也说明参数少不等于推理快：10步采样时，默认组中位时延349.9760 ms、LoRA 407.1434 ms；5步分别为201.2357 ms和233.6729 ms。这个差异不能未经 profiling 就归咎于某个模块。LoRA 5步只有测速，没有对应闭环成功率。

**证据：** `reports/phase1_baseline_parameters.json`、`reports/phase1_lora_parameters.json`；`scripts/train_entry.py`；SmolVLA 的 `_get_default_peft_targets`。

**继续追问：** LoRA 保存后为什么还需要基础模型？如果换基础模型版本或 processor，能直接复用 adapter 吗？

### 7. 训练和验证怎样划分？如何避免泄漏？

**答题要点：**

- 根据 LIBERO 的任务自然语言定义匹配数据任务；对每任务候选轨迹用 `seed+task_id` 的随机数流打乱，选 20 条训练、5 条验证，并保存绝对 episode 编号。
- 同一轨迹相邻帧高度相关，随机拆帧会让训练与验证过于相似，因此按整条轨迹划分。
- 输入官方训练器的每任务 episode 顺序为训练在前、验证在后，`eval_split=0.2`；官方按每任务末尾比例持出，与 manifest 对应。不要只看比例，要核对实际 episode 顺序。
- 状态/动作的 mean、std、分位数等只用训练轨迹重新计算。原统计和本地统计分别保存，并保留哈希；验证轨迹不参与这些统计。
- 固定数据副本的 episode 元数据缺少 `tasks` 字段时，脚本由帧级 `episode_index/task_index` 和任务表补全该字段，备份原文件并保存哈希。只修改本地副本，没有篡改动作或上传修改后的远程数据。
- 当前预防的是轨迹交叉和验证统计参与训练等泄漏。这不等于证明基础模型预训练从未见过相关数据，也不等于新任务或全新场景泛化。

**证据：** `scripts/prepare_data.py` 的 `repair_episode_tasks`、划分与统计逻辑；`configs/data_selection.json`；`reports/episode_metadata_repair.json`；`reports/train_only_stats.json`；官方 `datasets/factory.py`。

**继续追问：** 为什么使用 `std(ddof=0)`？图像统计为什么保留？本配置视觉归一化为 `IDENTITY`；状态/动作统计仍需严格隔离验证集。

### 8. 这是公平对照吗？可以说“LoRA 导致成功率变化”吗？

**答题要点：**

- 两组固定基础权重、任务和轨迹、batch size、更新步数、训练 seed、动作参数和评测设置。
- 但默认组学习率为 `1e-4`、衰减末值 `2.5e-6`；LoRA 组为 `1e-3`、末值 `1e-4`。这是两个实用训练配方的比较，并非严格只改变 LoRA 一个变量的消融。
- 当前观测是默认组 76/100、LoRA 63/100，相差 13 个百分点；准确表述是“本次配方与预算下观察到差异”。不能直接推导“LoRA 本身导致下降 13 个百分点”或“LoRA 在所有任务上更差”；任务 2 的实测结果为默认组 18/20、LoRA 19/20。
- `train_entry.py` 在模型/processor 构建后重置 Torch 随机种子，减轻 adapter 初始化对后续采样随机数流的影响；不能因此保证两条优化轨迹完全相同。
- 5000 次更新是两组相同的更新预算，不保证相同计算量或墙钟时间。应同时报告参数、显存、时间与闭环效果。
- 初版只有一个训练 seed。若要归因 LoRA 本身，应补充受控学习率实验和多个训练 seed；若调参，应使用验证数据，避免反复根据最终闭环结果挑配置。
- 与配方比较不同，已完成的采样对照固定同一个 baseline 最终 checkpoint，只改 `num_steps=10→5`，因此可以讨论当前环境与设置下这一推理干预的效果，但仍不能据此推广到其他模型、训练种子或真机。

**证据：** `scripts/run_experiment.py` 的两条学习率分支；`scripts/train_entry.py` 的 seed 重置；各实验保存的命令和配置。

**继续追问：** 结果没有提升是否项目失败？不是。可靠地解释资源—效果权衡、找到问题并设计下一步实验，同样能体现实验能力。

### 9. 离线 loss 和任务成功率有什么区别？

**答题要点：**

- 离线 loss 在固定示范观测上衡量速度场预测误差，不是直接执行后的动作误差，更不是任务完成概率。
- 闭环中策略会改变环境，后续观测由自身动作决定，误差可能累积或进入示范很少覆盖的状态。夹爪时机、接触和任务阈值也可能让较小的预测误差造成失败。
- 正式配置每任务 20 回合，共 5 个任务、每方法 100 回合；使用相同评测 seed 2000、固定初始状态、hard reset、每批 4 个环境等设置，逐任务与总体一起分析。
- 已完成的默认组成功 76 回合、LoRA 成功 63 回合；这是按环境任务成功条件记录的结果，不是从离线 loss 换算得到。训练为 BF16，正式闭环推理使用 FP16 autocast。
- 读取 `eval_info.json` 中实际回合与成功统计，核对视频。100 回合没有训练 seed 的重复，不能据此宣称覆盖训练不确定性或达到完整 LIBERO 成绩。
- 20 步 smoke 训练后的一次 40 步短回合，只是流程检查。它的失败不能用于判断正式策略上限。

**证据：** `scripts/run_experiment.py` 的训练验证与评测命令；`runs/phase1_*_eval/eval_info.json`；`scripts/summarize.py`。

**继续追问：** loss 降了但成功率没升，你会先看接口/统计、逐任务表现、训练覆盖、失败视频还是延长训练？先基于证据定位，不只做一项盲目调参。

### 10. 为什么用了两张 GPU，却不能写多卡分布式训练？

**答题要点：**

- `launch.py` 为两种方法启动独立的 `run_experiment.py`；每个进程用 `CUDA_VISIBLE_DEVICES` 选择一张物理卡，各自维护模型、优化器和数据流。
- DDP 是多个进程共同训练同一个模型，通过梯度通信同步参数。当前实验没有跨两卡的梯度同步，也不能把 batch size 16 说成两卡联合有效 batch 32。
- `start_new_session=True` 用于分离子进程和 SSH 会话，不能保证服务器关闭后继续训练。
- checkpoint 不只是权重，还包含配置、processor 和训练状态。新 phase 默认从基础权重开始；真正续训需要官方 resume 配置、优化器、scheduler、随机状态与训练步数。

**证据：** `scripts/launch.py`；`scripts/run_experiment.py`；`README.md` 的恢复说明。

**继续追问：** 当前两卡的收益是同时比较两个实验的周转时间，不是证明同一个模型获得双卡加速。

### 11. 推理耗时如何测量？把采样从 10 步降到 5 步能说明什么？

**答题要点：**

- `benchmark_latency.py` 只接受含训练配置、processor、权重和有效训练步数的 checkpoint，避免误测基础模型；LoRA 通过官方工厂加载基础模型和 adapter。
- 使用一个固定验证观测，不把真实动作输入策略；在 inference mode 与显式 autocast 下先预热（脚本默认BF16，本次受控采样对照用 `--precision fp16`），再在每次测量前后同步 GPU，报告原始时间、均值、中位数和 P95。
- 训练 BF16、闭环 FP16、主延迟 FP16 必须分别说明。旧 BF16 与新 FP16 的测速不能合并成一组“同精度对照”。已完成的五步采样对照固定同一 baseline checkpoint、FP16和GPU 0；每种采样设置分别在 `[10,5]` 与 `[5,10]` 两种顺序下预热10次、测量50次，再取两个顺序中位数的中位数。
- 本脚本测量同步动作块生成，包含该调用内部的初始噪声生成；不包含数据解码、processor 预处理、仿真渲染、动作后处理或任务完成耗时。它只覆盖一个观测，不能代表所有输入的端到端时延分布。
- `num_steps=10/5` 对照固定其他动作参数。少一半积分步数不意味着总耗时必然恰好减半，因为条件编码等存在固定成本。
- 不能只测速度而推断任务效果。本项目额外完成了保持相同初始状态的100回合闭环对照：10步和5步均成功76次；逐任务由17/12/18/18/11次变为15/12/18/18/13次。它由 `run_sampling_ablation.py` 编排，不是测速脚本自动完成的。
- 实测中位动作块时延由349.9760329723358 ms降至201.2357208877802 ms，下降42.5001423%。符合预先定义的工程筛选规则；任务0下降10个百分点，刚好达到允许上限。应说“在当前固定子集观察到该权衡”，不能说统计无损、非劣效或完整任务速度提高42.50%。
- LoRA 的FP16微基准为10步407.1434 ms、5步233.6729 ms，但没有运行LoRA 5步闭环，不能将其63/100的10步成功率移植到5步设置。
- autocast 控制运算精度，不意味着所有保存参数都已转换为对应的 BF16 或 FP16；报告会记录参数 dtype。

**证据：** `scripts/benchmark_latency.py`、`scripts/run_sampling_ablation.py`、`configs/sampling_ablation.json`、`reports/sampling5_baseline.json`；SmolVLA 的 `predict_action_chunk`、`sample_actions`。实际时延输出文件名为 `reports/<name>_latency_<时间戳>_<标识>.json`，避免覆盖旧结果。

**继续追问：** 为什么 CPU 计时前后需要 CUDA 同步？没有同步时可能只测到异步任务提交时间。

### 12. 训练或闭环失败时怎么排查？当前能得出什么结论？

**答题要点：**

按下面顺序定位，不要直接归因于“模型不行”：

| 现象 | 先检查的证据 | 可检验的下一步 |
|---|---|---|
| 数据加载失败 | task 字段、路径、固定 revision、视频解码日志 | 用一个 episode 检查读取、图像和形状；核对元数据修复记录 |
| 训练 NaN 或 loss 异常 | 状态/动作有限性、标准差、学习率、梯度、padding mask | 看训练统计与 batch，缩小试跑范围，分别验证数据和数值设置 |
| 显存不足 | 参数冻结、batch、图片、峰值 allocated/reserved、其他进程 | 先确认占用来源，再降低 batch 或调整配置；记录有效 batch 变化 |
| 仿真无法渲染 | EGL、assets、reset/step 报错 | 查看 `reports/libero_probe.json` 与 `setup_libero.py` 的检查范围 |
| 能运行但动作明显异常 | 相机键/方向、状态姿态、归一化/反归一化、动作语义、模型与 processor 配对 | 固定一个样本和初始状态，逐段检查输入与输出尺度 |
| 抓取接近但总失败 | 逐任务视频、夹爪时机、动作分块、示范覆盖 | 先提出一个原因，再只改一个执行或训练因素作对照 |
| 进度似乎停止 | 状态文件、最新 progress、完整日志、GPU/磁盘 | 区分训练、保存、验证、评测、进程退出，不只看 GPU 利用率 |

当前两组正式训练和三组闭环评测均已完成，可以引用默认组76/100、LoRA 63/100、同基线五步采样76/100及正式参数、时延和审计记录。`reports/FAILURE_ANALYSIS.md` 对三例 baseline 失败只给出观察与待验证假设，不能将其当成全部24个失败的原因统计。已完成的是 `num_steps` 积分采样步数对照；失败分析提出的 `n_action_steps` 反馈频率验证仍是未来方案，两者不能混淆。三组共300条原始视频、展示来源哈希映射和初值/seed/结果标签已核查。所有结论均限定在单训练种子、五任务子集及当前设置，不能推广到真实机器人或整个基准。

**证据：** `scripts/status.py`；`reports/phase1_*_status.json`；`runs/phase1_*_eval/eval_info.json`；`runs/sampling5_baseline_eval/eval_info.json`；`reports/FAILURE_ANALYSIS.md`；`reports/sampling5_baseline.json`；`reports/evaluation_audit.json`；`reports/media_manifest.json`；对应 progress 与日志；`scripts/setup_libero.py`；`reports/data_audit.json`。

**继续追问：** 没有真机经验会怎样补？可以说明已掌握仿真接口与数据流程，希望在实习中补采集、标定和部署；不要把智能车或仿真经历写成机械臂真机经历。

## 面试前最后自查

- 能否不用背稿，指着一条轨迹说明输入、目标、数据划分和统计来源？
- 能否画出模型的数据流，写对 Flow Matching 的时间方向和速度目标？
- 能否从参数名证明默认组与 LoRA 组实际训练了什么？
- 能否指出学习率混杂、单训练 seed、任务子集、测速范围这些限制？
- 能否找到一个当前结果的原始文件，而不是只展示汇总截图？
- 能否手写并解释一个简化 Attention、训练循环或张量维度检查？会调用模型不等于具备 Coding 基础。

## 简历草稿模板

以下模板包含已完成的训练、参数、闭环和受控采样结果，个人职责仍需按自己实际承担、理解和验证的部分调整。时延必须保留单观测、FP16、模型动作块生成的范围限定：

> 基于开源 SmolVLA、LeRobot 与 LIBERO，完成固定五任务的数据检查与整轨划分，使用 100 条训练、25 条留出轨迹，对默认动作专家微调与 LoRA 各训练 5000 步；在每任务 20 回合的闭环评测中分别取得 76/100 与 63/100 成功，可训练参数分别为 99,880,992 与 1,485,312，整理逐任务结果、失败案例与可复现配置，并说明两组学习率不同的配方比较边界。
>
> 对同一基线权重开展10→5步的Flow Matching采样对照，在100个固定初值回合中总成功率均为76%，固定单观测、FP16、GPU 0的动作块生成中位时延由349.98 ms降至201.24 ms（下降42.50%）；同时披露任务0下降10个百分点、任务4上升10个百分点的逐任务变化，保留300条评测视频与初值/结果/来源映射，不将工程筛选通过表述为统计无损。

## 固定版本资料

- [本项目采用的 LeRobot 固定提交](https://github.com/huggingface/lerobot/tree/f9e9d739ea7055b282ea1b165e8785412a674693)
- [SmolVLA 官方介绍](https://huggingface.co/docs/lerobot/smolvla)
- [官方 PEFT/LoRA 使用指南](https://huggingface.co/docs/lerobot/peft_training)
- [LIBERO 数据与仿真评测指南](https://huggingface.co/docs/lerobot/libero)

网页文档会更新。面试涉及具体字段和实现时，以固定提交、当前脚本与保存的真实配置为准。
