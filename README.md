# SmolVLA × LeRobot × LIBERO

机器人操作策略适配、闭环评测与采样计算权衡。基于开源SmolVLA，在固定LIBERO-Spatial五任务上完成默认动作专家和LoRA微调，并用相同权重、初值与seed评估10→5步采样。

**[视频与成果展示](https://zxyzhk.github.io/smolvla-libero-lab/)** · [300回合完整评测](https://zxyzhk.github.io/smolvla-libero-lab/technical.html) · [完整报告](EXPERIMENT_REPORT.md) · [简历条目](RESUME.md) · [面试精读](INTERVIEW_GUIDE.md)

主页按“同场景仿真视频 → 核心结果 → 技术贡献”展示；逐任务表、完整图库、训练曲线与证据保留在二级页。

## 实测结果

| 配置 | 可训练参数 | 闭环成功率 | Wilson 95%区间 |
|---|---:|---:|---|
| 默认动作专家，10步采样 | 99,880,992 | 76/100 | 66.8%–83.3% |
| 基于官方PEFT的LoRA r32配置，10步采样 | 1,485,312 | 63/100 | 53.2%–71.8% |
| 同一默认专家权重，5步采样 | 复用基线 | 76/100 | 66.8%–83.3% |

两种训练配方各5000次更新、batch16、训练seed1000；两张RTX4090分别独立训练，约26.3/26.9分钟，没有DDP。100条训练与25条留出轨迹按整轨划分，共11,939/2,962帧。评测每任务20个相同初值，共100回合/设置。

LoRA可训练参数减少98.51%，当前配方成功率低13个百分点。两组学习率不同，是实用配方比较，不能单独归因于LoRA。基线10→5步的FP16动作块生成中位时延349.98→201.24 ms，下降42.50%，总体成功率76%→76%，通过预设工程筛选；逐任务仍存在变化，不能称统计无损。完整对照见[报告](EXPERIMENT_REPORT.md)和[原始对照](docs/data/sampling_ablation_result.json)。

这是一项单训练种子、固定五任务子集的工程实验；不代表完整LIBERO或真机表现。训练采用BF16，闭环及主时延比较采用FP16。时延是固定单观测、batch1的动作块生成微基准，包含内部图像准备，排除外部processor和仿真；视频按20Hz仿真控制时间播放，不含推理等待。

## 项目工作

- 适配双相机图像、8维状态与7维动作；修复源episode元数据，保留原始备份与哈希，仅由训练轨迹重算状态/动作统计。
- 复现默认动作专家及基于官方PEFT的LoRA配置，记录实际可训练参数、训练指标、checkpoint重载和归一化buffer审计。
- 修复同步向量环境早终止后提前消耗后续初值的问题；用不同回合长度的回归验证，并追踪每个真实reset的初值与seed。
- 在结果揭晓前登记10→5采样对照；固定checkpoint、FP16、相同初值与执行动作数，保留相反测试顺序的原始计时。
- 保存300个闭环视频与逐回合结果，核对唯一源路径、成功标签、初值、seed及源/展示文件哈希；从真实失败回合提出待验证假设。

模型、LoRA、Flow Matching和基准来自上游；本项目的小trick是受控工程对照。具体贡献、固定版本与许可见[来源说明](ACKNOWLEDGEMENTS.md)。

## 代码与证据

| 入口 | 内容 |
|---|---|
| [RUNBOOK](RUNBOOK.md) | 从依赖、数据到训练、对照、审计与页面导出的实际命令 |
| [EXPERIMENT_REPORT](EXPERIMENT_REPORT.md) | 成功率、逐任务结果、配对变化、时延、成本与结论边界 |
| [RESUME](RESUME.md) | 使用实测结果的简历条目与60秒介绍 |
| [INTERVIEW_GUIDE](INTERVIEW_GUIDE.md) | 四天精读路线、核心技术栈与12个追问 |
| [结果JSON](docs/data/results.json) / [任务CSV](docs/data/task_metrics.csv) | 页面使用的真实指标与区间 |
| [配对回合CSV](docs/data/paired_outcomes.csv) | 同初值10/5步的逐回合成功标记 |
| [完整审计](docs/data/evaluation_audit.json) / [视频清单](docs/data/media_manifest.json) | 300回合的初值、标签、路径与哈希 |
| [失败分析](docs/data/FAILURE_ANALYSIS.md) | 三个真实失败案例的观察、假设与后续验证计划 |
| [补丁](patches/) / [回归验证](tests/test_sync_initial_states.py) | 同步环境冻结、初值追踪、完整视频与终止帧 |

预训练权重及完整示范数据按固定版本下载，不包含在Git中。原始回合视频、训练日志和指标同时保存在本地证据包与服务器，网页中的展示副本可直接播放；模型checkpoint保留在服务器。`docs/`是无外部前端依赖的静态页面，由GitHub Pages发布。

## 复现

先按[RUNBOOK](RUNBOOK.md)配置Python、系统渲染依赖、固定版本模型/数据与LIBERO资源。脚本拒绝覆盖已存在实验目录；完整重跑优先使用新的项目目录。当前推理对照与审计脚本使用固定的phase1实验名称，其他phase需要同步修改相应配置与路径。

```bash
source env.sh
python scripts/apply_patches.py
python tests/test_sync_initial_states.py
python scripts/launch.py --phase phase1 --steps 5000 --batch-size 16
```

正式评测后依次执行采样对照、源审计、媒体导出、完整媒体审计和展示导出，具体顺序与日志路径以RUNBOOK为准。最终文档由`scripts/write_delivery.py`根据已审计结果生成，未完成时拒绝生成最终结论。
