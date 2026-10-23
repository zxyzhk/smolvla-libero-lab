#!/usr/bin/env python
"""Summarize saved evidence using the actual LeRobot metrics/eval schemas."""
import argparse
import json
import math
import os
import re
from pathlib import Path


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_json(path, warnings):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        warnings.append(f"暂不能读取 {path.name}：{error}")
        return None


def read_metrics(path, warnings):
    rows = []
    if path.exists():
        for index, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
            except ValueError:
                warnings.append(f"忽略 {path.name} 第{index}行未完成或无效的JSON记录")
    return rows


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def fmt(value, precision=3):
    return f"{value:.{precision}f}" if numeric(value) else "—"


def integer(value):
    return f"{int(value):,}" if numeric(value) and int(value) == value else "—"


def flag(command, name):
    for index, value in enumerate(command or []):
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
        if value == name and index + 1 < len(command):
            return command[index + 1]
    return None


def success(record):
    """Read Wilson bounds already in percent. Never fabricate missing intervals."""
    if not isinstance(record, dict):
        record = {}
    result = {key: record.get(key) for key in
              ("n_success", "n_episodes", "pc_success", "pc_success_ci95")}
    flags = record.get("metrics", {}).get("successes")
    if ((result["n_success"] is None or result["n_episodes"] is None)
            and isinstance(flags, list)
            and all(isinstance(v, (bool, int, float)) and v in (0, 1) for v in flags)):
        result.update(n_success=sum(bool(v) for v in flags), n_episodes=len(flags),
                      counts_source="recorded per-episode success flags")
    k, n = result["n_success"], result["n_episodes"]
    if not (numeric(k) and numeric(n) and int(k) == k and int(n) == n and 0 <= k <= n):
        result["n_success"] = result["n_episodes"] = None
    elif n > 0 and not numeric(result["pc_success"]):
        result["pc_success"] = 100 * k / n
        result["rate_source"] = "computed from recorded counts"
    if not numeric(result["pc_success"]):
        result["pc_success"] = None
    ci = result["pc_success_ci95"]
    if not (isinstance(ci, (list, tuple)) and len(ci) == 2
            and all(numeric(v) for v in ci) and 0 <= ci[0] <= ci[1] <= 100):
        result["pc_success_ci95"] = None
    return result


def success_text(result):
    if result["n_episodes"] == 0:
        return "0回合，暂无成功率"
    if result["n_episodes"] is None and result["pc_success"] is None:
        return "尚无评测结果"
    counts = (f"{integer(result['n_success'])}/{integer(result['n_episodes'])}"
              if result["n_episodes"] is not None else "回合数未记录")
    rate = f"{result['pc_success']:.1f}%" if numeric(result["pc_success"]) else "成功率未记录"
    return f"{counts}，{rate}"


def ci_text(result):
    ci = result["pc_success_ci95"]
    return f"[{ci[0]:.1f}%, {ci[1]:.1f}%]" if ci is not None else "—（未记录）"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", default="phase1")
    parser.add_argument("--output-dir", type=Path, help="Default reports/<phase>; relative to project root")
    args = parser.parse_args()
    if args.phase in (".", "..") or not re.fullmatch(r"[A-Za-z0-9_.-]+", args.phase):
        parser.error("phase must be a simple experiment name")
    root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
    reports = root / "reports"
    output = args.output_dir or reports / args.phase
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    warnings = []
    manifest = read_json(root / "configs/data_selection.json", warnings) or {}
    audit = read_json(reports / "data_audit.json", warnings) or {}
    expected_ids = manifest.get("benchmark", {}).get("task_ids", [0, 1, 2, 3, 4])
    task_names = {t["benchmark_task_id"]: t.get("instruction", "")
                  for t in manifest.get("selection", {}).get("tasks", [])}
    combined, plot_rows = {}, {}
    for method in ("baseline", "lora"):
        name = f"{args.phase}_{method}"
        paths = {
            "status": reports / f"{name}_status.json",
            "parameters": reports / f"{name}_parameters.json",
            "progress": reports / f"{name}_progress.json",
            "metrics": reports / f"{name}_metrics.jsonl",
            "evaluation": root / "runs" / f"{name}_eval" / "eval_info.json",
            "log": root / "logs" / f"{name}.log",
        }
        state = read_json(paths["status"], warnings) or {"stage": "not_started"}
        params = read_json(paths["parameters"], warnings) or {}
        progress = read_json(paths["progress"], warnings) or {}
        evaluation = read_json(paths["evaluation"], warnings)
        rows = read_metrics(paths["metrics"], warnings)
        validation = []
        if paths["log"].exists():
            for step, loss in re.findall(r"step (\d+): eval_loss=([\deE+.-]+)",
                                         paths["log"].read_text(errors="replace")):
                value = float(loss)
                if math.isfinite(value):
                    validation.append({"step": int(step), "eval_loss": value})
        if progress and not any(row.get("step") == progress.get("step") for row in rows):
            rows.append(progress)
        rows.sort(key=lambda row: row.get("step") if numeric(row.get("step")) else -1)
        usable = [row for row in rows if numeric(row.get("step"))
                  and numeric(row.get("metrics", {}).get("loss"))]
        plot_rows[method] = usable
        allocated = [r["peak_allocated_gb"] for r in rows if numeric(r.get("peak_allocated_gb"))]
        reserved = [r["peak_reserved_gb"] for r in rows if numeric(r.get("peak_reserved_gb"))]
        overall = success(evaluation.get("overall", evaluation.get("aggregated", {}))
                          if isinstance(evaluation, dict) else {})
        per_task = [{"task_group": t.get("task_group"), "task_id": t.get("task_id"), "success": success(t)}
                    for t in (evaluation.get("per_task", []) if isinstance(evaluation, dict) else [])]
        smoke = bool(state.get("smoke_only")) or args.phase.lower().startswith("smoke")
        elapsed = state.get("training_wall_s")
        complete_time = numeric(elapsed)
        if not complete_time:
            elapsed = progress.get("training_elapsed_s")
        trainable, total = params.get("trainable_parameters"), params.get("total_parameters")
        summary = {
            "kind": "pipeline_check" if smoke else "formal_initial_experiment",
            "latest_recorded_update": progress.get("step"), "planned_updates": state.get("steps"),
            "trainable_parameters": trainable, "total_parameters": total,
            "trainable_fraction": trainable / total if numeric(trainable) and numeric(total) and total > 0 else None,
            "max_logged_allocated_gib": max(allocated) if allocated else None,
            "max_logged_reserved_gib": max(reserved) if reserved else None,
            "first_logged_loss": usable[0]["metrics"]["loss"] if usable else None,
            "last_logged_loss": usable[-1]["metrics"]["loss"] if usable else None,
            "loss_log_points": len(usable), "training_time_s": elapsed,
            "validation_loss": validation,
            "training_time_complete_subprocess": complete_time,
            "evaluation": {"overall": overall, "per_task": per_task,
                           "episodes_per_task": flag(state.get("eval_command"), "--eval.n_episodes"),
                           "episode_length_override": flag(state.get("eval_command"), "--env.episode_length")},
        }
        combined[method] = {"status": state, "parameters": params, "progress": progress,
                            "evaluation": evaluation, "summary": summary,
                            "evidence_paths": {key: str(path.relative_to(root)) for key, path in paths.items()}}
        latency_paths = sorted(reports.glob(f"{name}_latency_*.json"))
        combined[method]["latency"] = read_json(latency_paths[-1], warnings) if latency_paths else None
        if latency_paths:
            combined[method]["evidence_paths"]["latency"] = str(latency_paths[-1].relative_to(root))

    smoke = any(value["summary"]["kind"] == "pipeline_check" for value in combined.values())
    lines = [f"# SmolVLA 实验记录：{args.phase}", "",
             "范围：LIBERO-Spatial固定任务子集；每张GPU运行一个独立实验，不是双卡分布式训练。", "",
             ("**本阶段仅为流程检查。短训练、少回合或截断回合用于验证数据→训练→保存→评测链路，不能用于判断策略能力或方法优劣。**"
              if smoke else "本阶段是单训练种子的初版正式实验；只有已写入的评测结果才作为任务表现证据。"), "",
             "默认动作专家微调与LoRA使用各自学习率，属于实用配方比较；差异不能全部归因于LoRA。", "",
             "## 训练证据", "",
             "| 方法 | 状态 | 记录更新/计划 | 可训练/总参数 | 末次训练loss | 记录分配/保留显存GiB | 训练耗时秒 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for method, value in combined.items():
        s = value["summary"]
        suffix = "" if s["training_time_complete_subprocess"] else "（记录中）"
        lines.append(f"| {method} | {value['status'].get('stage', 'unknown')} | "
                     f"{integer(s['latest_recorded_update'])}/{integer(s['planned_updates'])} | "
                     f"{integer(s['trainable_parameters'])}/{integer(s['total_parameters'])} | "
                     f"{fmt(s['last_logged_loss'], 5)} | "
                     f"{fmt(s['max_logged_allocated_gib'], 2)}/{fmt(s['max_logged_reserved_gib'], 2)} | "
                     f"{fmt(s['training_time_s'], 1)}{suffix} |")
    lines += ["", "显存为实际CUDA日志中记录值的最大值。上游可在更新中重置峰值计数，因此不能保证是完整训练的全局峰值，也不同于nvidia-smi占用。",
              "训练loss可能为日志窗口均值，不是留出验证loss，也不等于闭环任务成功率。"]
    for method, value in combined.items():
        validation = value["summary"]["validation_loss"]
        if validation:
            latest = validation[-1]
            lines.append(f"- {method} 最近一次留出验证：step {latest['step']}，"
                         f"eval_loss={latest['eval_loss']:.4f}；仅记录有限验证样本的离线损失。")
    if any(plot_rows.values()):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axis = plt.subplots(figsize=(8, 4.5))
        for method, rows in plot_rows.items():
            if rows:
                axis.plot([r["step"] for r in rows], [r["metrics"]["loss"] for r in rows], label=method)
        axis.set(xlabel="Recorded optimizer updates", ylabel="Logged training loss",
                 title=f"{args.phase}: training loss (not task success)")
        axis.grid(alpha=.2)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output / "training_loss.png", dpi=160)
        plt.close(fig)
        lines += ["", "![训练损失](training_loss.png)"]

    lines += ["", "## 闭环评测证据", "",
              "| 方法 | 实验性质 | 整体成功数/回合数及成功率 | 已记录95% Wilson区间 | 每任务计划回合数 |",
              "|---|---|---|---|---:|"]
    truncation_notes = []
    for method, value in combined.items():
        s = value["summary"]
        ev = s["evaluation"]
        kind = "仅流程检查" if s["kind"] == "pipeline_check" else "初版正式实验"
        lines.append(f"| {method} | {kind} | {success_text(ev['overall'])} | "
                     f"{ci_text(ev['overall'])} | {ev['episodes_per_task'] or '—'} |")
        if ev["episode_length_override"]:
            truncation_notes.append(f"- {method} 使用episode_length={ev['episode_length_override']}的回合截断。")
    lines += [""] + truncation_notes + ["", "### 逐任务结果", "",
              "| 任务ID | 指令 | baseline成功数/回合数及成功率 | baseline 95%区间 | LoRA成功数/回合数及成功率 | LoRA 95%区间 |",
              "|---|---|---|---|---|---|"]
    observed_ids = {t["task_id"] for v in combined.values() for t in v["summary"]["evaluation"]["per_task"]
                    if isinstance(t.get("task_id"), int)}
    for task_id in sorted(set(expected_ids) | observed_ids):
        task_success = {}
        for method, value in combined.items():
            matches = [t for t in value["summary"]["evaluation"]["per_task"]
                       if t["task_id"] == task_id and t["task_group"] in (None, "libero_spatial")]
            task_success[method] = matches[0]["success"] if matches else success({})
        instruction = task_names.get(task_id, "—").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {task_id} | {instruction} | {success_text(task_success['baseline'])} | "
                     f"{ci_text(task_success['baseline'])} | {success_text(task_success['lora'])} | "
                     f"{ci_text(task_success['lora'])} |")
    lines += ["", "区间直接读取eval_info.json的pc_success_ci95，数值已是百分比；缺失时不补造。"
              "整体区间对应已测回合，不能代替跨任务、跨训练种子的不确定性分析。", "",
              "## 动作生成时延", "",
              "仅测单条留出观测、batch=1、BF16 autocast 下的模型动作序列生成。排除预处理、仿真和渲染；不代表整个回合耗时。", "",
              "| 方法 | 采样步数 | 测量次数 | 中位数ms | P95 ms |",
              "|---|---:|---:|---:|---:|"]
    for method, value in combined.items():
        latency = value.get("latency")
        if not latency:
            lines.append(f"| {method} | — | — | — | — |")
            continue
        for row in latency.get("rows", []):
            lines.append(f"| {method} | {integer(row.get('num_steps'))} | "
                         f"{integer(row.get('measurements'))} | {fmt(row.get('median_ms'), 2)} | "
                         f"{fmt(row.get('p95_ms'), 2)} |")
    lines += ["", "测速不包含5步采样的闭环成功率，因此不能据此判断减少采样步数后任务效果是否保持。", "",
              "## 数据与流程检查", ""]
    if audit:
        lines += [f"- 数据源版本：{audit.get('source', {}).get('revision', '未记录')}。",
                  f"- 训练帧/留出帧：{integer(audit.get('training_frames'))}/{integer(audit.get('heldout_frames'))}；"
                  f"状态/动作维度：{integer(audit.get('state_dimensions'))}/{integer(audit.get('action_dimensions'))}。",
                  f"- 状态、动作、时间戳有限性：{audit.get('finite_state_action_timestamp', '未记录')}。",
                  f"- 归一化范围：{audit.get('normalization', {}).get('scope', '未记录')}。"]
    else:
        lines.append("- 尚未找到数据审计记录。")
    for method, value in combined.items():
        lines.append(f"- {method}：参数记录{'存在' if value['parameters'] else '缺失'}，"
                     f"训练loss记录{value['summary']['loss_log_points']}点，"
                     f"闭环评测文件{'存在' if value['evaluation'] is not None else '尚未生成'}，"
                     f"流程状态为 {value['status'].get('stage', 'unknown')}。")
        if value["status"].get("error"):
            lines.append(f"  最近错误：{value['status']['error']}")
        evidence = root / value["evidence_paths"]["evaluation"]
        if evidence.exists():
            lines.append(f"  [评测原始记录]({os.path.relpath(evidence, output)})。")
    lines += ["", "## 解释边界", "",
              "- 缺失值显示为‘—’或‘尚无评测结果’，不等同于零；未结束的实验不预测成功率。",
              "- 本报告对应选定任务、当前训练种子和已记录回合，不是完整LIBERO成绩。",
              "- 参数、显存和耗时来自本次运行，不代表其他硬件或训练规模。",
              "- 当前没有真机部署、Sim2Real、大规模预训练或分布式训练结果。",
              "- 仅改变num_steps的扩展实验应固定n_action_steps、数据和评测初始状态。"]
    if warnings:
        lines += ["", "## 文件读取提示", ""] + [f"- {warning}" for warning in warnings]
    combined["_metadata"] = {"phase": args.phase, "contains_pipeline_check": smoke,
                             "report_directory": str(output), "warnings": warnings,
                             "confidence_interval_source": "Recorded Wilson 95% interval in percent"}
    (output / "experiment_summary.json").write_text(
        json.dumps(clean_json(combined), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (output / "EXPERIMENT_REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved phase-specific report: {output / 'EXPERIMENT_REPORT.md'}")


if __name__ == "__main__":
    main()
