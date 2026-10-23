# SmolVLA + LeRobot + LIBERO 复现实操

本文说明从新环境准备到导出展示页的执行顺序，不记录某次实验的最新成绩。结果以 `runs/*_eval/eval_info.json`、训练日志及审计报告为准。以下命令在服务器项目根目录执行；首次运行使用 `phase1`，已有实验的目录与日志应保留。

## 1. 固定环境与实验协议

两张 RTX 4090 分别承担独立单卡训练，不进行跨卡梯度同步。已验证环境为 Python 3.12.3、Torch 2.8.0+cu128、Transformers 5.5.4；完整依赖见 `bootstrap.sh`、`constraints.txt` 和 `reports/environment.freeze.txt`。

| 项目 | 固定设置 |
|---|---|
| 默认服务器根目录 | `/root/autodl-tmp/smolvla_lab` |
| LeRobot | `f9e9d739ea7055b282ea1b165e8785412a674693`，另应用本项目三个补丁 |
| SmolVLA 基础权重 | `lerobot/smolvla_base@5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6` |
| 视觉语言模型 | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct@7b375e1b73b11138ff12fe22c8f2822d8fe03467` |
| 数据 | `lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` |
| 任务与轨迹划分 | LIBERO-Spatial 任务 0–4；每任务 20 条训练、5 条留出；划分 seed 1000 |
| 正式训练 | 每配方 5000 步，batch size 16，训练 seed 1000，BF16 mixed precision |
| 正式闭环评测 | 每任务 20 回合，共 100 回合；初始状态 0–19、seed `2000 + initial_state_id`、同步环境 batch size 4、FP16 autocast |
| 动作设置 | `chunk_size=50`、`n_action_steps=10`；主配方 `num_steps=10` |

默认组冻结视觉编码器，训练动作专家和状态投影，学习率 `1e-4`；LoRA 采用官方PEFT实现，rank 32、alpha 32、学习率 `1e-3`。两组学习率不同，属于实用训练配方比较；默认组也不应称为整个 VLA 全量微调。实际参数名称和数量由训练入口保存到 `reports/*_parameters.json`。

## 2. 新环境安装与上游补丁

先将本仓库放到目标目录。更换目录时，在运行 `bootstrap.sh` 和 `source env.sh` **之前**设置 `PROJECT_ROOT`；新目录应包含本仓库的脚本、配置、补丁和测试文件。

```bash
cd /root/autodl-tmp/smolvla_lab
export PROJECT_ROOT="$PWD"
# BASE_PYTHON 默认是 /root/miniconda3/bin/python；必要时改为本机 Python 3.12。
export BASE_PYTHON=/root/miniconda3/bin/python

apt-get update
apt-get install -y ffmpeg libegl1 libegl-mesa0 libgl1-mesa-dri libosmesa6 libglib2.0-0
bash bootstrap.sh
source env.sh
```

`bootstrap.sh` 创建 `.venv`、检出固定 LeRobot 提交、按 constraints 安装依赖，并自动调用 `apply_patches.py`。已经安装依赖的环境也可以单独执行下面的补丁检查；已应用的补丁通过反向检查识别，不会重复叠加。

```bash
python scripts/apply_patches.py
python scripts/setup_libero.py --configure-only
python tests/test_sync_initial_states.py
```

三个补丁均作用于 `vendor/lerobot` 的本地工作区，不修改上游提交：

| 补丁 | 目的 |
|---|---|
| `patches/libero_sync_freeze.patch` | 同步向量环境中，冻结已经结束的 lane，避免其他 lane 尚未结束时再次 autoreset、消耗后续 LIBERO 初始状态 |
| `patches/libero_initial_state_trace.patch` | 记录实际执行的 `LIBERO_INIT task_id=... initial_state_id=... seed=...`，供配对审计 |
| `patches/eval_complete_videos.patch` | 保留每任务全部评测回合的视频，并保留初始观测及最后一次动作后的画面 |

补丁来源和 SHA256 保存在 `reports/upstream_patches.json`。`tests/test_sync_initial_states.py` 是不依赖真实仿真画面的回归测试：四条 lane 故意在不同步数结束，五批次应恰好覆盖初始状态 0–19，已结束 lane 不再多执行动作。它不能替代后续对真实 `LIBERO_INIT` 日志的审计。

以后每次重新登录，先进入项目目录并执行 `source env.sh`。该文件配置项目缓存、EGL、虚拟环境、LIBERO 配置路径和本地日志模式。`HF_ENDPOINT` 默认使用镜像，也可以在 source 前显式设置；模型和数据仍按固定提交下载并记录来源。

## 3. 准备模型、数据与无显示器仿真

先配置 LIBERO 路径，再准备资源、模型和数据，最后执行真实 EGL reset/step 检查：

```bash
python scripts/setup_libero.py --configure-only

curl -L --fail --retry 3 \
  https://codeload.github.com/Lifelong-Robot-Learning/LIBERO/tar.gz/8f1084e3132a39270c3a13ebe37270a43ece2a01 \
  -o vendor/libero-master.tar.gz

python scripts/prepare_models.py
python scripts/prepare_data.py --max-workers 4
python scripts/setup_libero.py
```

`setup_libero.py` 从该压缩包提取 assets，并为安装的 LIBERO 包配置路径。资源来源和本地目录见 `reports/libero_assets.json`，真实渲染探针见 `reports/libero_probe.json`。应在两者正常后开始训练。

`prepare_data.py` 下载视频版数据到 `data/libero`，通过任务指令匹配 benchmark 任务，并按整条轨迹划分训练和留出集合。固定数据缺少 episode 级 `tasks` 列，脚本从帧级任务索引回填，只修复本地副本，原 parquet 备份到 `reports/source_episode_metadata/`。它还将原始统计备份到 `reports/source_stats.json`，只用训练轨迹重新计算状态/动作统计并覆盖本地 `meta/stats.json`；留出轨迹不参与统计。图像统计保持原值，SmolVLA 图像归一化采用 `IDENTITY`。划分、任务匹配、8 维状态 / 7 维动作、有限性和样例图都写入数据报告。

如果数据已经完整下载，只需重新做适配与审计：

```bash
python scripts/prepare_data.py --skip-download
```

`--skip-download` 仍会查询数据提交号；它不是完全离线模式。改变数据版本或划分时应使用新的项目目录，避免已有训练记录与重写后的 manifest、归一化统计混用。迁移项目时也要检查 `configs/data_selection.json` 中的数据绝对路径，以及基础模型、processor 和 adapter 配置中的本地路径。

## 4. 在正式结果产生前登记推理对照

受控对照采用 `configs/sampling_ablation.json`：固定使用默认动作专家的 5000 步 checkpoint，不根据哪一组成功率更高来挑模型；**唯一推理干预是 `policy.num_steps: 10 → 5`**。`chunk_size=50`、`n_action_steps=10`、FP16、任务、初始状态、seed 和同步 batch size 均保持不变，不重复训练。

在新的复现目录里，先审核该配置，再在查看正式闭环结果之前登记协议哈希。下列命令独占创建登记文件，已有记录不会覆盖；已经看过正式结果时，不应填写“未查看结果”。

```bash
python - <<'PYREGISTER'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
root = Path(os.environ["PROJECT_ROOT"])
protocol = root / "configs/sampling_ablation.json"
if any((root / f"runs/phase1_{method}_eval/eval_info.json").exists()
       for method in ("baseline", "lora")):
    raise RuntimeError("正式评测文件已存在；不要补写为事前登记")
record = {
    "registered_at_utc": datetime.now(timezone.utc).isoformat(),
    "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
    "formal_closed_loop_results_inspected": False,
}
with (root / "reports/sampling_preregistration.json").open("x") as handle:
    json.dump(record, handle, indent=2)
PYREGISTER
```

协议预设工程筛选条件为：基线成功率至少 20%，中位动作生成时延下降至少 20%，总成功率下降不超过 5 个百分点、任一任务下降不超过 10 个百分点。无论是否通过都公布结果，不追加扫描其他步数来挑选较好结果。这些规则不构成统计非劣效检验。

## 5. 流程检查与两组正式训练

先确认 GPU 空闲，再运行短流程检查：

```bash
python scripts/launch.py --phase smoke --steps 20 --batch-size 16 --smoke
python scripts/status.py
python scripts/summarize.py --phase smoke
```

两张卡各训练 20 步，随后各执行任务 0 的一个 40 步短回合。它只验证读数据、反向传播、保存/重载模型和仿真执行；不能用短回合的成败评价模型能力。等待两组 smoke 都结束、检查日志无误后，再开始正式训练：

```bash
python scripts/launch.py --phase phase1 --steps 5000 --batch-size 16 &&
(
  set -o noclobber
  nohup python scripts/finish_phase.py --phase phase1 --skip-latency \
    > logs/phase1_finalize.log 2>&1 < /dev/null &
)
```

上述 finalizer 启动命令仅用于新的 `phase1`，同名 finalizer 不要重复启动。`launch.py` 已将训练放入独立进程：GPU 0 为 `phase1_baseline`，GPU 1 为 `phase1_lora`；各自在训练结束后自动用最终 checkpoint 做 5 任务 × 20 回合的 10 步 FP16 闭环评测。无需另行启动相同评测。SSH 断开通常不影响这些进程，释放或关闭服务器会中断工作。

训练中间和结束时保存 checkpoint，正式配置在 2500、5000 步保存；每 1000 步计算有限数量的留出样本离线损失。训练 loss / 留出损失与闭环任务成功率分别衡量不同内容。`training_wall_s` 是训练子进程墙钟耗时；显存报告是已保存日志中的 CUDA allocated 最大值，不等同整机占用或显卡总峰值。

这里让 finalizer 使用 `--skip-latency`：它等待两组训练与评测完成，核对 checkpoint 并生成 phase 专属报告，为后续受控 FP16 测速留出空闲 GPU。默认不加该参数时，finalizer 调用的是 `benchmark_latency.py` 的默认 **BF16** 测速；这些结果只能作为补充，不能与 FP16 闭环结果拼成主比较。不要让 finalizer 测速、正式评测和其他 GPU 作业同时争用同一张卡。

## 6. 运行 10 → 5 步推理对照与 FP16 测速

只有在 `phase1` 两组已完成、finalizer 已结束且 GPU 0 空闲后，才进入对照评测。当前脚本也会等待 `reports/phase1_finalize.json` 达到 `completed` 或 `completed_with_errors`，再检查两组实验确实为 `completed`；它最多等待 6 小时，不会替你停止其他进程。

**必须将输出写到 `logs/sampling_ablation.log`**：脚本会读取这个路径和 `logs/phase1_baseline.log`，逐任务比对实际初始状态。下面的 `noclobber` 保护已有日志；若已运行过，请按第 9 节建立新的实验，而不是清空旧日志重试。

```bash
(
  set -o noclobber
  nohup python scripts/run_sampling_ablation.py \
    > logs/sampling_ablation.log 2>&1 < /dev/null &
)
```

`run_sampling_ablation.py` **没有 CLI 参数**，读取上述 JSON 配置。它拒绝覆盖 `runs/sampling5_baseline_eval`，复用基线权重执行 5 步 FP16 闭环，并要求每任务实际记录初始状态 0–19 和对应 seed。之后在物理 GPU 0 上，依次为 baseline、LoRA 测试 FP16 动作块生成耗时：每组按 `10,5` 与 `5,10` 两种顺序各测一遍，每设置预热 10 次、测量 50 次。

结果写到 `reports/sampling5_baseline.json`；原始测速分别保存为带时间戳的 `reports/controlled_<method>_latency_*.json`，不覆盖此前采样。主时延汇总使用两种顺序各自中位数的中位数；展示页的 P95 采用两次顺序测试中较大值。LoRA 的 5 步设置只在这里做模型测速，没有对应的 LoRA 5 步闭环结果。

以下是独立、手工测速的实际 CLI 示例；受控脚本已自动完成上述工作时，不需要重复执行。若额外运行，请用不同 `--name` 区分补充测量，避免 exporter 把它们混入 `controlled_*` 主比较：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_latency.py \
  --checkpoint runs/phase1_baseline/checkpoints/last/pretrained_model \
  --name extra_baseline_fp16 --precision fp16 \
  --warmup 10 --measurements 50 --sampling-steps 10 5

# 仅作额外精度对照；与主 FP16 结果分开报告。
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_latency.py \
  --checkpoint runs/phase1_baseline/checkpoints/last/pretrained_model \
  --name extra_baseline_bf16 --precision bf16 \
  --warmup 10 --measurements 50 --sampling-steps 10 5
```

测速使用同一留出观测、batch size 1、固定随机种子并同步 CUDA。计时范围是 `predict_action_chunk`，包含模型内部图像准备和初始噪声，排除外部 processor、数据解码、仿真、渲染与动作后处理；它不代表完整机器人墙钟响应时延。

## 7. 审计结果，再生成媒体与展示数据

依赖顺序是：**三组完整闭环结果与原视频 → 评测审计 → 展示视频/封面 → 报告与曲线 → 展示页 JSON**。这些命令依次执行，不应在上游文件仍写入时提前导出：

```bash
(
  set -e
  python scripts/audit_evaluations.py
  python scripts/prepare_media.py \
    --experiments phase1_baseline phase1_lora sampling5_baseline
  python scripts/summarize.py --phase phase1
  python scripts/audit_evaluations.py --require-media
  python scripts/export_showcase.py
  python scripts/write_delivery.py
)
```

- `audit_evaluations.py` 当前固定检查这三个实验及其日志。它要求每组 100 回合、每任务初始状态与 seed 完整配对、逐回合布尔成功标记与计数一致、Wilson 95% 区间重算一致，且所有 300 个原始视频存在、非空、路径唯一并精确属于对应实验/任务/回合。媒体导出后再加 `--require-media`，核对全部源/展示视频哈希、标签、初值与seed。审计通过才写入 `reports/evaluation_audit.json`；任一组未完成时命令会报错，不应用旧审计文件声称新结果已通过。
- `prepare_media.py` 依赖 `ffmpeg` / `ffprobe` 和原评测视频。唯一 CLI 是 `--experiments` 列表，默认就是上述三组。它保留原视频，另存展示视频及 JPG 封面到 `docs/assets/videos/<experiment>/`；转换按 **20 Hz 仿真控制时间** 播放，不含推理等待，核对转换前后帧数，并以源文件 SHA256 缓存已处理媒体。清单为 `reports/media_manifest.json`。不要把展示播放速度当作实时机器人速度。
- `summarize.py --phase phase1` 生成 `reports/phase1/EXPERIMENT_REPORT.md`、汇总 JSON 和有真实 loss 时的 `training_loss.png`；smoke 使用独立 `reports/smoke/`，不会覆盖 phase1。也可加 `--output-dir reports/phase1_review` 导出另一份报告。
- `export_showcase.py` 没有 CLI 参数，按脚本所在项目目录读取证据，生成 `docs/data/results.json`、`protocol.json`、`task_metrics.csv` 及证据副本，并复制训练曲线。可选 `reports/interpretation.json` 提供结论与失败案例，`reports/FAILURE_ANALYSIS.md` 提供分析文档；缺少它们仍可导出。展示页失败案例图片/视频引用的文件需事先存在于 `docs/` 对应位置，exporter 不会代为创建这些分析媒体。

导出脚本会更新展示数据、媒体清单和派生文件；原始 `runs/`、日志与报告证据应保留。页面完成状态由三组评测、采样对照状态、300 条媒体清单和通过的评测审计共同判断。部分数据可以预览，但不能据此宣布完整实验复核完成。

静态页面不需要前端安装或构建，在项目目录启动本地预览后打开 `http://127.0.0.1:8000`：

```bash
python -m http.server 8000 --directory docs --bind 127.0.0.1
```

若在远程服务器启动，需要自行转发该端口。不要直接以 `file://` 打开页面，否则浏览器可能阻止读取 JSON。

## 8. 查看进度与主要产物

```bash
python scripts/status.py
tail -n 80 logs/phase1_baseline.log
tail -n 80 logs/phase1_lora.log
tail -n 80 logs/phase1_finalize.log
tail -n 80 logs/sampling_ablation.log
python scripts/summarize.py --phase phase1
df -h "$PROJECT_ROOT"
```

`status.py` 用于查看实验驱动阶段与 GPU 情况；采样对照另看 `reports/sampling5_baseline.json` 和日志。`completed` 表示对应命令正常结束，模型表现仍需查看 `eval_info.json`；`completed_with_errors` 的 finalizer 还需检查其 `errors` 数组。

| 路径 | 用途 |
|---|---|
| `reports/runtime.json`、`environment.freeze.txt`、`lerobot_commit.txt`、`upstream_patches.json` | 实际环境、上游版本与三个本地补丁 |
| `reports/models.json` | 模型固定版本、权重哈希和本地路径修改 |
| `configs/data_selection.json`、`reports/data_audit.json` | 任务、轨迹划分、数据版本、统计与检查依据 |
| `reports/episode_metadata_repair.json`、`source_episode_metadata/` | 元数据本地修复记录与原始文件 |
| `reports/train_only_stats.json`、`source_stats.json` | 训练统计与原始全量统计备份 |
| `reports/*_parameters.json`、`*_metrics.jsonl`、`*_progress.json`、`*_status.json` | 参数、loss、日志显存、训练进度与流程状态 |
| `runs/phase1_*/checkpoints/last/pretrained_model/` | 最终模型或 LoRA adapter、配置和 processor |
| `runs/phase1_baseline_eval/`、`phase1_lora_eval/`、`sampling5_baseline_eval/` | 真实闭环 `eval_info.json` 与原始视频 |
| `configs/sampling_ablation.json`、`reports/sampling_preregistration.json`、`sampling5_baseline.json` | 预设协议、事前登记和实际对照状态/结果 |
| `reports/controlled_*_latency_*.json` | 主比较的 FP16 原始测速与配置 |
| `reports/evaluation_audit.json`、`media_manifest.json` | 评测证据与媒体转换审计 |
| `reports/phase1/`、`docs/data/`、`docs/assets/` | 派生报告、静态展示数据与媒体 |

LoRA adapter 仍依赖基础模型；移动 checkpoint 时要保留其基础权重、adapter 配置、processor 和路径可用性。复制项目不会自动重写训练配置或 eval_info 内的绝对路径；媒体转换前应核对这些源视频路径。

## 9. 重跑、续训与中断处理

`launch.py` 的新训练拒绝覆盖已有 run 或 PID 记录。只想重新训练时，可以使用新的 phase：

```bash
python scripts/launch.py --phase phase2 --steps 10000 --batch-size 16 &&
(
  set -o noclobber
  nohup python scripts/finish_phase.py --phase phase2 --skip-latency \
    > logs/phase2_finalize.log 2>&1 < /dev/null &
)
python scripts/summarize.py --phase phase2
```

这会从基础模型重新训练，不会接着 phase1 的 5000 步继续。真正续训需使用固定版本 LeRobot 的 `--resume=true` 与 checkpoint `pretrained_model/train_config.json`，并保留 `training_state`、优化器、scheduler 和随机状态，核对恢复后的总步数和学习率计划；本项目 `launch.py` 没有 `--resume` 参数。

**`--phase phase2` 不会自动切换完整的展示流水线。** 当前 `run_sampling_ablation.py` 固定等待 `phase1_finalize.json`、检查 phase1 两组状态、读取 `phase1_baseline.log` 与 `sampling_ablation.log`，测速也固定使用 phase1 checkpoint 和 `controlled_*` 名称；即使修改协议 JSON，5 步评测命令及其他协议字段也不会全部自动跟随。`audit_evaluations.py`、`export_showcase.py` 同样固定 phase1 和 `sampling5_baseline`。重跑完整流程时优先使用独立项目目录；如在同目录接入新 phase，须先修改这些脚本中的名称/路径、协议配置、日志名、审计映射和 exporter，并为对照设置新 `experiment_id`，不能假定它们支持不存在的 `--phase` CLI。`prepare_media.py --experiments ...` 支持传新名称，但当前仍要求每组 5 任务 × 20 回合。

仅评测中断且 checkpoint 完整时，`launch.py --phase <原phase> --eval-only` 可以重试未完成的评测；smoke 还要带 `--smoke`。它不会重训，但会追加原日志。**正式评测有严格的初始状态日志审计，追加多次评测 trace 会使审计不再满足每任务恰好 20 条。** 因此不要将这个快捷重试当作新的正式对照：先确认所有原进程退出并保存失败现场，再采用新的评测目录、独立日志和相应配置/审计映射。脚本拒绝对已经 `completed` 的实验这样重试。

不要删除旧 `runs/`、清空日志或改写既有登记来绕过保护。增加训练 seed、换数据划分、换任务或精度，都应更新实际配置、独立命名并重新审计；只换实验名称不会产生新的随机种子证据。
