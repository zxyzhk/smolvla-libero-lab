#!/usr/bin/env python3
"""Write final report and CV material only after the complete evidence audit."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/data"


def read(path):
    return json.loads(path.read_text())


def main():
    data = read(DATA / "results.json")
    if not data["status"]["complete"]:
        raise RuntimeError("Final deliverables require completed evaluation, timings, media and audit")
    experiments = {e["id"]: e for e in data["experiments"]}
    base, lora, treatment = [experiments[n] for n in ("phase1_baseline", "phase1_lora", "sampling5_baseline")]
    ablation = read(DATA / "sampling_ablation_result.json")
    c = ablation["comparison"]
    reduction = 100 * (1-lora["training"]["trainable_parameters"]/base["training"]["trainable_parameters"])
    rate = lambda e: e["evaluation"]["pc_success"]
    ci = lambda e: "[" + ", ".join(f"{v:.1f}%" for v in e["evaluation"]["pc_success_ci95"]) + "]"
    rows = []
    for e in (base, lora, treatment):
        cost = e["training"]
        train = "复用基线权重" if e is treatment else f"{cost['wall_s']/60:.1f} min"
        rows.append(f"| {e['label']} | {cost['trainable_parameters']:,} | {train} | {rate(e):.0f}% ({e['evaluation']['n_success']}/100) | {ci(e)} |")
    tasks = ["| 任务 | 默认10步 | LoRA10步 | 同权重5步 |", "|---|---:|---:|---:|"]
    for i in range(5):
        counts = [e["evaluation"]["per_task"][i]["n_success"] for e in (base, lora, treatment)]
        tasks.append(f"| {i} | " + " | ".join(f"{k}/20" for k in counts) + " |")
    timings = ["| 权重 | 积分步数 | 中位动作块生成时延 | 保守P95 |", "|---|---:|---:|---:|"]
    for e in (base, lora):
        weight_name = "默认动作专家权重" if e is base else "LoRA r32权重"
        for t in e["latency"]:
            timings.append(f"| {weight_name} | {t['num_steps']} | {t['median_ms']:.2f} ms | {t['p95_ms']:.2f} ms |")
    reference = read(DATA / "phase1_baseline_eval.json")
    changed = ablation["evaluation"]
    transitions = {"both_success": 0, "only_10_success": 0, "only_5_success": 0, "both_failure": 0}
    with (DATA / "paired_outcomes.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["task_id", "initial_state_id", "seed", "success_10_steps", "success_5_steps"])
        for a, b in zip(reference["per_task"], changed["per_task"]):
            assert a["task_id"] == b["task_id"]
            for i, (x, y) in enumerate(zip(a["metrics"]["successes"], b["metrics"]["successes"])):
                writer.writerow([a["task_id"], i, 2000+i, x, y])
                key = ("both_success" if x and y else "only_10_success" if x else "only_5_success" if y else "both_failure")
                transitions[key] += 1
    screening = "通过" if c["engineering_screen_passed"] else "未通过"
    report = f"""# SmolVLA × LeRobot × LIBERO 实验报告

[展示页](https://zxyzhk.github.io/smolvla-libero-lab/) · [代码](https://github.com/zxyzhk/smolvla-libero-lab) · [操作文档](RUNBOOK.md) · [面试精读](INTERVIEW_GUIDE.md)

## 结果与范围

在 LIBERO-Spatial 固定任务0–4上，默认动作专家微调成功率为 **{rate(base):.0f}%**，基于官方PEFT的LoRA配置为 **{rate(lora):.0f}%**。相同基线权重将积分采样步数10→5后为 **{rate(treatment):.0f}%**，中位动作块生成时延下降 **{c['latency_reduction_percent']:.2f}%**，{screening}预先定义的工程筛选。模型、数据及既有算法均来自开源项目，贡献范围见[来源说明](ACKNOWLEDGEMENTS.md)。

| 配置 | 可训练参数 | 训练进程耗时 | 闭环成功率 | Wilson 95%区间 |
|---|---:|---:|---:|---|
{chr(10).join(rows)}

两种训练配方各5000次更新、batch16、seed1000，分别使用一张RTX4090；没有DDP。默认学习率1e-4、LoRA1e-3，因此此比较不能单独识别LoRA的因果影响。LoRA可训练参数减少{reduction:.2f}%，当前预算的总体成功率低{rate(base)-rate(lora):.0f}个百分点；不能推出所有任务上都更差或增加训练预算后仍如此。

## 数据和执行接口

- 固定版本 `lerobot/libero`：每任务20条训练、5条留出，共100/25条轨迹、11,939/2,962帧；按整轨划分。状态/动作统计仅从训练轨迹重算。
- 源episode元数据缺少tasks列，以帧级task_index恢复；保留原始备份与哈希。
- 主视角与腕部RGB各256×256；状态8维（末端位置3、轴角3、夹爪2），动作7维（相对6DoF、夹爪）。chunk_size50，每次执行前10个动作后重观测。
- 训练使用BF16；闭环评测使用CUDA autocast FP16。留出轨迹损失采用最多160个样本的局部验证，不能替代闭环成功率。
- 三组评测均5任务×20回合，seed2000，逐任务实际初值0–19与seed2000–2019均在日志中核对。
- 修复同步向量环境早终止回合自动重置、提前消耗后续初值的问题；使用不同回合长度的回归验证。另保留所有20回合视频及终止帧。

## 逐任务结果

{chr(10).join(tasks)}

任务原文、逐任务区间及完整逐回合布尔结果见[协议](docs/data/protocol.json)、[结果CSV](docs/data/task_metrics.csv)与[配对回合CSV](docs/data/paired_outcomes.csv)。同一初值两种设置的结果：共同成功{transitions['both_success']}，仅10步成功{transitions['only_10_success']}，仅5步成功{transitions['only_5_success']}，共同失败{transitions['both_failure']}。配对指相同初值与评测seed，运行轨迹及随机数消耗仍可因行为分叉而不同。

## 受控采样对照与测速

唯一干预是num_steps10→5；基线checkpoint固定5000步，n_action_steps始终10。协议及哈希在结果揭晓前保存；未追加更多采样步数挑选结果。

{chr(10).join(timings)}

上述时延均为同一物理GPU0、FP16、batch1、同一个留出观测，10次预热、每设置50次计时，按10/5与5/10两个顺序复测；中位数取两个顺序各自中位数的中位数，P95取较大者。包含predict_action_chunk内部图像准备与初始噪声，排除外部processor/tokenization、解码、仿真、渲染及动作后处理。它不是完整机器人响应时延。原预注册协议中的“excludes preprocessing”在此精确指外部processor；内部图像准备始终在实测范围内，没有改变计时边界。LoRA5步仅测速，没有做闭环效果评测。初版finalizer的BF16测速作为补充保留在服务器，未混入本表。

预设工程筛选：参考成功率≥20%、时延下降≥20%、总成功率降幅≤5个百分点、任一任务降幅≤10个百分点。实际总降幅{c['overall_success_drop_pp']:.0f}个百分点；逐任务降幅为 `{c['per_task_success_drop_pp']}`，总判定为**{screening}**。这不是统计无损或非劣效证明。

## 资源、视频与失败分析

训练日志每20步采样一次。已记录的CUDA allocated最大值：默认{base['training']['max_logged_allocated_gib']:.2f} GiB、LoRA{lora['training']['max_logged_allocated_gib']:.2f} GiB，不是整次训练的保证峰值，也不是显卡进程总占用。训练进程耗时包含加载等开销，不能直接作为纯计算吞吐。

全部300个原始评测视频保存在服务器与本地证据包；300个可播放副本及海报在展示页和本仓库保存。原视频80fps标签被修正为20Hz仿真控制时间，转换前后帧数一致，播放不含推理等待。清单保存原始/展示视频SHA256、帧数、结果、初值与seed；审计核对300个唯一原始路径及对应展示文件哈希和标签。

[三例失败分析](docs/data/FAILURE_ANALYSIS.md)根据真实失败回合抽样图描述现象；未把抓取、反馈或碰撞假设当作已证实原因，也没有把三个例子推广为全部失败的分类。下一步可单独改变n_action_steps检验更频繁反馈，尚未运行。

## 结论的边界

仅一个训练种子、五个固定任务；不代表完整LIBERO、全新场景泛化或真机部署。公开示范与仿真初值可能相关，预训练数据重叠未排除。Wilson区间是已测回合的描述，不替代跨训练种子和任务的统计分析。五步采样是既有采样器的工程对照，不是原创模型。

所有数字由保存记录生成：[完整结果](docs/data/results.json)、[采样对照](docs/data/sampling_ablation_result.json)、[审计](docs/data/evaluation_audit.json)、[视频清单](docs/data/media_manifest.json)。复现入口见RUNBOOK。
"""
    (ROOT / "EXPERIMENT_REPORT.md").write_text(report)
    conclusion = (f"成功率{rate(base):.0f}%→{rate(treatment):.0f}%，通过预设工程筛选" if c["engineering_screen_passed"]
                  else f"成功率{rate(base):.0f}%→{rate(treatment):.0f}%，如实记录未通过的工程筛选条件")
    resume = f"""# 可用项目经历与面试口径

## 项目名称

SmolVLA机器人操作策略适配与闭环评测｜PyTorch、LeRobot、LoRA、LIBERO、MuJoCo

## 简历条目

- 基于SmolVLA与LeRobot搭建机器人模仿学习训练—仿真闭环：适配双相机、8维状态与7维动作，按整轨划分100条训练/25条留出示范，重算训练集归一化统计；在LIBERO-Spatial固定5任务、100回合中取得{rate(base):.0f}%成功率。
- 完成默认动作专家与LoRA r32配方的各5000步对照，可训练参数99.88M→1.49M（减少{reduction:.2f}%），成功率{rate(base):.0f}%/{rate(lora):.0f}%；修复同步向量环境终止后提前消耗初值的问题，保存实际初值、逐任务指标与300个完整回合视频，建立结果审计和展示页。
- 固定权重、FP16、相同初值与seed，评估Flow Matching采样10→5步的性能—计算权衡：单观测batch1动作块生成中位时延下降{c['latency_reduction_percent']:.1f}%，{conclusion}；公开预设协议、逐回合结果、原始计时与失败案例。

## 60秒介绍

我做的是SmolVLA的少样本适配和可复查评测。重点工作是整理机器人轨迹与归一化、理解动作块与重观测的关系、复现默认专家及LoRA，并解决评测初始状态被早结束回合提前消耗的问题。默认专家在固定五任务100回合中成功{rate(base):.0f}次。LoRA参数更少但当前配方成功率较低。我随后固定同一checkpoint测试五步采样，实际得到{conclusion}，中位动作块生成时延下降{c['latency_reduction_percent']:.1f}%。我保留了所有回合的视频和指标，区分观察与原因假设。

## 必须能解释的证据

- 为什么state8/action7，控制量的坐标与相对动作语义，图像和状态如何进入策略。
- chunk_size50、n_action_steps10、num_steps10分别控制什么；少采样和更频繁重观测是两个不同变量。
- Flow Matching训练目标和Euler积分；LoRA实际训练哪些参数，为什么参数少不等于推理更快。
- 为什么不能把LoRA配方差异说成严格消融；为什么训练loss、仿真视频播放速度都不能当作闭环性能。
- 初值冻结补丁如何触发、修复前后回归结果，以及300回合的标签/路径/哈希如何交叉核对。

本项目是开源模型适配、工程评测与小规模实验，不应写为原创VLA、完整LIBERO SOTA、双卡DDP或真机部署。具体问答见[INTERVIEW_GUIDE](INTERVIEW_GUIDE.md)，全部结果见[报告](EXPERIMENT_REPORT.md)。
"""
    (ROOT / "RESUME.md").write_text(resume)
    print(json.dumps({"report": "EXPERIMENT_REPORT.md", "resume": "RESUME.md", "paired_transitions": transitions}, ensure_ascii=False))


if __name__ == "__main__":
    main()
