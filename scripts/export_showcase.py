#!/usr/bin/env python
"""Export a static portfolio from recorded evidence, preserving missing results."""
import csv
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DATA = DOCS / "data"
DATA.mkdir(parents=True, exist_ok=True)


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def export(source, name):
    if source.exists():
        shutil.copy2(source, DATA / name)
        return {"label": name, "href": f"data/{name}"}


def main():
    manifest = read(ROOT / "configs/data_selection.json", {})
    audit = read(ROOT / "reports/data_audit.json", {})
    media = read(ROOT / "reports/media_manifest.json", {"videos": []})
    instructions = {t["benchmark_task_id"]: t["instruction"]
                    for t in manifest.get("selection", {}).get("tasks", [])}
    ablation = read(ROOT / "reports/sampling5_baseline.json", {})
    registered_latency_paths = []
    experiments, evidence = [], []
    for method, label in (("baseline", "默认动作专家 · 10步采样"), ("lora", "LoRA r32 · 10步采样")):
        name = f"phase1_{method}"
        status = read(ROOT / f"reports/{name}_status.json", {})
        progress = read(ROOT / f"reports/{name}_progress.json", {})
        params = read(ROOT / f"reports/{name}_parameters.json", {})
        evaluation = read(ROOT / f"runs/{name}_eval/eval_info.json", {})
        memory = []
        for line in (ROOT / f"reports/{name}_metrics.jsonl").read_text().splitlines() if (ROOT / f"reports/{name}_metrics.jsonl").exists() else []:
            try:
                memory.append(json.loads(line)["peak_allocated_gb"])
            except (ValueError, KeyError):
                pass
        latency_paths = [ROOT / path for path in ablation.get("latency_reports", {}).get(method, [])]
        latencies = []
        if len(latency_paths) == 2 and all(path.exists() for path in latency_paths):
            import statistics
            reports = [read(path) for path in latency_paths]
            assert all(report["precision"]["autocast"] == "cuda_float16"
                       and report["cuda_visible_devices"] == "0" and report["checkpoint_step"] == 5000
                       and sorted(row["num_steps"] for row in report["rows"]) == [5, 10]
                       for report in reports), "Latency evidence differs from the controlled protocol"
            registered_latency_paths += latency_paths
            rows = [row for report in reports for row in report["rows"]]
            for n in (10, 5):
                matching = [r for r in rows if r["num_steps"] == n]
                latencies.append({"num_steps": n, "median_ms": statistics.median(r["median_ms"] for r in matching),
                                  "p95_ms": max(r["p95_ms"] for r in matching), "precision": "FP16"})
        per_task = [{key: t.get(key) for key in ("task_id", "n_success", "n_episodes", "pc_success", "pc_success_ci95")}
                    | {"instruction": instructions.get(t["task_id"], "")}
                    for t in evaluation.get("per_task", [])]
        experiments.append({"id": name, "label": label, "kind": "formal_recipe_comparison",
                            "training": {"steps": progress.get("step"), "target_steps": status.get("steps"),
                                         "trainable_parameters": params.get("trainable_parameters"),
                                         "total_parameters": params.get("total_parameters"),
                                         "max_logged_allocated_gib": max(memory) if memory else None,
                                         "wall_s": status.get("training_wall_s")},
                            "evaluation": evaluation.get("overall", {}) | {"per_task": per_task},
                            "latency": latencies,
                            "videos": [v for v in media["videos"] if v["experiment"] == name],
                            "notes": ["状态：" + status.get("stage", "not_started"),
                                      "训练与验证按完整轨迹划分；显存仅指已记录CUDA allocated最大值。",
                                      "闭环评测FP16。时延若有记录，同为FP16、固定单观测、batch=1；P95取两种测试顺序中较大值。",
                                      "本行成功率来自10步采样。LoRA的5步设置只做模型测速；默认动作专家5步的闭环结果单独列为推理对照。"]})
        for source, output in ((ROOT / f"runs/{name}_eval/eval_info.json", f"{name}_eval.json"),
                               (ROOT / f"reports/{name}_metrics.jsonl", f"{name}_training.jsonl"),
                               (ROOT / f"reports/{name}_parameters.json", f"{name}_parameters.json")):
            item = export(source, output)
            if item:
                evidence.append(item)
    protocol = read(ROOT / "configs/sampling_ablation.json", {})
    comparison = ablation.get("comparison")
    if ablation.get("evaluation"):
        baseline = experiments[0]
        ev = ablation["evaluation"]
        experiments.append({"id": "sampling5_baseline", "label": "相同基线权重 · 5步采样", "kind": "controlled_inference_ablation",
                            "training": baseline["training"],
                            "evaluation": ev["overall"] | {"per_task": [
                                {key: t.get(key) for key in ("task_id", "n_success", "n_episodes", "pc_success", "pc_success_ci95")}
                                | {"instruction": instructions.get(t["task_id"], "")} for t in ev["per_task"]]},
                            "latency": [r for r in baseline["latency"] if r["num_steps"] == 5],
                            "videos": [v for v in media["videos"] if v["experiment"] == "sampling5_baseline"],
                            "notes": ["复用baseline最终权重，不重复训练；唯一干预为积分采样步数10→5。", "初始状态配对检查：" + str(ablation.get("initial_state_audit", {}).get("passed"))]})
    conclusions = [("默认动作专家与LoRA各完成5000步训练。" if all(e["training"].get("steps") == 5000 for e in experiments[:2]) else "初版计划每组训练5000步，当前进度见表格。") + "两组使用相同基础模型、100条训练轨迹和25条留出轨迹；学习率随配方变化，不能把差异全部归因于LoRA。"]
    if experiments[0]["training"].get("trainable_parameters") and experiments[1]["training"].get("trainable_parameters"):
        a, b = (e["training"]["trainable_parameters"] for e in experiments[:2])
        conclusions.append(f"可训练参数：{a:,} → {b:,}，减少{100*(1-b/a):.2f}%。这不意味着显存或推理速度等比例改善。")
    if all(e["evaluation"].get("n_episodes") == 100 for e in experiments[:2]):
        a, b = (e["evaluation"] for e in experiments[:2])
        conclusions.append(f"固定五任务评测各100回合：默认动作专家{a['n_success']}/100（{a['pc_success']:.1f}%），LoRA {b['n_success']}/100（{b['pc_success']:.1f}%）。区间及逐任务结果完整公开。")
    ablation_conclusion = "等待正式闭环结果。预设筛选：中位时延至少下降20%，总成功率最多下降5个百分点、任一任务最多下降10个百分点，且基线至少20%成功。"
    if comparison:
        c = comparison
        conclusions.append(f"相同基线权重下，10→5步：成功率{c['reference_success_percent']:.1f}%→{c['treatment_success_percent']:.1f}%，中位动作块生成时延下降{c['latency_reduction_percent']:.1f}%。")
        ablation_conclusion = (f"成功率{c['reference_success_percent']:.1f}%→{c['treatment_success_percent']:.1f}%，中位动作块生成时延下降{c['latency_reduction_percent']:.1f}%。" +
                               ("通过预先定义的工程筛选规则；仍不能宣称统计上无损或非劣效。" if c["engineering_screen_passed"] else "未通过预先定义的全部工程筛选规则；不把减步数表述为无损加速。"))
        conclusions.append(ablation_conclusion)
    interpretation = read(ROOT / "reports/interpretation.json", read(ROOT / "configs/interpretation.json", {}))
    conclusions += interpretation.get("conclusions", [])
    evaluation_audit = read(ROOT / "reports/evaluation_audit.json", {})
    complete = (all(e["evaluation"].get("n_episodes") == 100 for e in experiments)
                and len(experiments) == 3 and ablation.get("stage") == "completed"
                and all(len(e["latency"]) == 2 for e in experiments[:2])
                and len(media["videos"]) == 300 and evaluation_audit.get("passed") is True
                and all((DOCS / video[key]).is_file() for video in media["videos"] for key in ("src", "poster"))
                and evaluation_audit.get("media_passed") is True)
    protocol_export = {"selection": manifest.get("selection"), "benchmark": manifest.get("benchmark"),
                       "normalization": manifest.get("normalization"), "sampling_ablation": protocol,
                       "data_revision": manifest.get("dataset", {}).get("revision")}
    (DATA / "protocol.json").write_text(json.dumps(protocol_export, indent=2, ensure_ascii=False))
    for source, output in ((ROOT / "configs/sampling_ablation.json", "sampling_protocol.json"),
                           (ROOT / "reports/sampling5_baseline.json", "sampling_ablation_result.json"),
                           (ROOT / "reports/sampling_preregistration.json", "sampling_preregistration.json"),
                           (ROOT / "reports/upstream_patches.json", "upstream_patches.json"),
                           (ROOT / "reports/evaluation_audit.json", "evaluation_audit.json"),
                           (ROOT / "reports/data_audit.json", "data_audit.json")):
        item = export(source, output)
        if item:
            evidence.append(item)
    for path in registered_latency_paths:
        item = export(path, path.name)
        if item:
            evidence.append(item)
    for name in ("models.json", "source_manifest.json", "train_only_stats.json",
                 "episode_metadata_repair.json", "libero_assets_verification.json",
                 "environment.freeze.txt", "phase1_baseline_checkpoint_audit.json",
                 "phase1_lora_checkpoint_audit.json", "checkpoint_hashes.json", "media_manifest.json"):
        item = export(ROOT / "reports" / name, name)
        if item:
            evidence.append(item)
    evidence.insert(0, {"label": "固定实验协议与轨迹划分", "href": "data/protocol.json"})
    if (ROOT / "reports/FAILURE_ANALYSIS.md").exists():
        evidence.append(export(ROOT / "reports/FAILURE_ANALYSIS.md", "FAILURE_ANALYSIS.md"))
    payload = {"project": {"title": "SmolVLA 机器人操作实验", "subtitle": "从示范数据适配、参数高效微调到闭环执行：用可复查的对照结果，评估任务表现与推理计算的权衡。",
                           "repository_url": "https://github.com/zxyzhk/smolvla-libero-lab", "updated_at": datetime.now(timezone.utc).isoformat()},
               "status": {"label": "实验结果已复核" if complete else "训练与闭环评测进行中", "complete": complete},
               "protocol": {"tasks": list(instructions), "train_episodes": len(manifest.get("selection", {}).get("train_episodes", [])),
                            "heldout_episodes": len(manifest.get("selection", {}).get("heldout_episodes", [])),
                            "train_frames": audit.get("training_frames"), "heldout_frames": audit.get("heldout_frames"), "seed": 1000,
                            "notes": ["LIBERO-Spatial固定任务0–4；每任务20条训练、5条留出轨迹。", "两张RTX 4090独立单卡训练；没有跨卡梯度同步。", "默认组与LoRA学习率不同，是实用配方比较。", "推理减步对照固定checkpoint、FP16、初始状态0–19、n_action_steps=10和seed=2000。", "同步环境终止冻结修复及全部初值日志均保留；视频按20Hz仿真控制时间播放，排除推理等待。"]},
               "experiments": experiments,
               "ablations": [{"id": "sampling_steps", "label": "10 → 5步：计算节省是否值得？", "status": ablation.get("stage", "waiting"),
                               "hypothesis": "使用同一最终基线权重，仅减少Flow Matching的Euler积分步数；同时检查100回合成功率和FP16动作生成时延。",
                               "conclusion": ablation_conclusion}],
               "conclusions": conclusions, "limitations": ["单个训练种子、固定五任务子集；不是完整LIBERO成绩，也不代表真机部署。", "仿真初始状态评测可能与公开示范初值相关；本实验不证明全新场景或预训练数据无重叠。", "Wilson区间对应已测回合，不能代替跨任务与训练种子的统计分析。", "时延是固定单观测、batch=1的predict_action_chunk微基准，包含模型内部图像准备；排除外部processor、仿真及完整机器人墙钟响应。", "小trick是既有采样器的受控工程对照，未提出原创模型或采样算法。"],
               "failure_cases": interpretation.get("failure_cases", []), "evidence": evidence}
    curve = ROOT / "reports/phase1/training_loss.png"
    if curve.exists():
        shutil.copy2(curve, DOCS / "assets/training_loss.png")
        payload["project"]["training_curve_src"] = "assets/training_loss.png"
    (DATA / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
    with (DATA / "task_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["experiment", "task_id", "successes", "episodes", "success_percent", "ci95_low_percent", "ci95_high_percent"])
        for experiment in experiments:
            for task in experiment["evaluation"].get("per_task", []):
                writer.writerow([experiment["id"], task["task_id"], task.get("n_success"), task.get("n_episodes"), task.get("pc_success"), *(task.get("pc_success_ci95") or [None, None])])
    (DOCS / ".nojekyll").touch()
    print(json.dumps({"experiments": len(experiments), "videos": len(media["videos"]), "complete": complete}, indent=2))


if __name__ == "__main__":
    main()
