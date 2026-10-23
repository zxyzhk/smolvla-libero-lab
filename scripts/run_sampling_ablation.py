#!/usr/bin/env python
"""Run the preregistered 10->5 step ablation after the existing phase finishes."""
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import psutil


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main():
    root = Path(os.environ["PROJECT_ROOT"])
    protocol_path = root / "configs/sampling_ablation.json"
    protocol = read(protocol_path)
    name = protocol["experiment_id"]
    destination = root / "reports" / f"{name}.json"
    prior = read(destination)
    if prior and (prior["stage"] == "completed" or psutil.pid_exists(prior["pid"])):
        raise RuntimeError("This ablation already completed or has a live process")
    state = {"id": name, "pid": os.getpid(), "stage": "waiting", "started_at_unix": time.time(),
             "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest()}

    def save():
        temp = destination.with_suffix(".json.tmp")
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(destination)

    def run(command):
        print("COMMAND", json.dumps(command), flush=True)
        subprocess.run(command, cwd=root, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0",
                                                 MUJOCO_EGL_DEVICE_ID="0"), check=True)

    save()
    try:
        deadline = time.monotonic() + 6 * 3600
        while True:
            finalizer = read(root / "reports/phase1_finalize.json")
            if finalizer and finalizer["stage"] in {"completed", "completed_with_errors"}:
                break
            if finalizer and finalizer["stage"] == "failed":
                raise RuntimeError(f"Phase finalizer failed: {finalizer.get('error')}")
            if time.monotonic() > deadline:
                raise TimeoutError("Waiting expired; phase1 processes were not stopped")
            time.sleep(20)
        for method in ("baseline", "lora"):
            s = read(root / f"reports/phase1_{method}_status.json")
            if not s or s["stage"] != "completed":
                raise RuntimeError(f"Missing completed phase1 {method}")
        checkpoint = root / protocol["checkpoint"]
        state["checkpoint"] = str(checkpoint)
        state["model_sha256"] = hashlib.file_digest((checkpoint / "model.safetensors").open("rb"), "sha256").hexdigest()
        output = root / "runs" / f"{name}_eval"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        command = [sys.executable, "-m", "lerobot.scripts.lerobot_eval",
                   f"--policy.path={checkpoint}", "--policy.device=cuda", "--policy.use_amp=true",
                   "--policy.num_steps=5", "--policy.n_action_steps=10", "--env.type=libero",
                   "--env.task=libero_spatial", "--env.task_ids=[0,1,2,3,4]",
                   "--env.observation_height=256", "--env.observation_width=256",
                   "--env.init_states=true", "--env.hard_reset=true", "--env.max_parallel_tasks=1",
                   "--eval.n_episodes=20", "--eval.batch_size=4", "--eval.use_async_envs=false",
                   "--seed=2000", f"--output_dir={output}"]
        state.update(stage="evaluating", eval_command=command)
        save()
        run(command)
        state["evaluation"] = read(output / "eval_info.json")
        if state["evaluation"]["overall"]["n_episodes"] != 100:
            raise RuntimeError("Incomplete ablation evaluation")
        traces = {}
        for label, path in (("reference", root / "logs/phase1_baseline.log"),
                            ("treatment", root / "logs/sampling_ablation.log")):
            rows = re.findall(r"LIBERO_INIT task_id=(\d+) initial_state_id=(\d+) seed=(\d+)",
                              path.read_text(errors="replace"))
            per_task = {task: [(int(i), int(s)) for t, i, s in rows if int(t) == task]
                        for task in range(5)}
            expected = [(i, 2000 + i) for i in range(20)]
            if any(values != expected for values in per_task.values()):
                raise RuntimeError(f"Actual initial-state trace differs from protocol: {label}")
            traces[label] = per_task
        state["initial_state_audit"] = {"passed": True, "traces": traces}
        state.update(stage="benchmarking")
        save()
        latency_reports = {}
        for method in ("baseline", "lora"):
            paths = []
            for order in ((10, 5), (5, 10)):
                before = set((root / "reports").glob(f"controlled_{method}_latency_*.json"))
                run([sys.executable, str(root / "scripts/benchmark_latency.py"),
                     "--checkpoint", str(root / f"runs/phase1_{method}/checkpoints/last/pretrained_model"),
                     "--name", f"controlled_{method}", "--precision", "fp16", "--warmup", "10",
                     "--measurements", "50", "--sampling-steps", *map(str, order)])
                created = set((root / "reports").glob(f"controlled_{method}_latency_*.json")) - before
                if len(created) != 1:
                    raise RuntimeError("Cannot identify unique latency report")
                paths += list(created)
            latency_reports[method] = [str(p.relative_to(root)) for p in paths]
        state["latency_reports"] = latency_reports
        by_steps = {n: [] for n in (10, 5)}
        for path in latency_reports["baseline"]:
            for row in read(root / path)["rows"]:
                by_steps[row["num_steps"]].append(row["median_ms"])
        medians = {n: statistics.median(v) for n, v in by_steps.items()}
        reference = read(root / protocol["reference_evaluation"])
        treatment = state["evaluation"]
        drops = {a["task_id"]: a["pc_success"] - b["pc_success"]
                 for a, b in zip(reference["per_task"], treatment["per_task"], strict=True)}
        overall_drop = reference["overall"]["pc_success"] - treatment["overall"]["pc_success"]
        latency_reduction = 100 * (1 - medians[5] / medians[10])
        rules = protocol["descriptive_acceptance_rules"]
        criteria = {
            "reference_success_is_informative": reference["overall"]["pc_success"] >= rules["minimum_reference_success_percent_for_retention_claim"],
            "latency_reduction_passed": latency_reduction >= rules["minimum_median_latency_reduction_percent"],
            "overall_success_drop_passed": overall_drop <= rules["maximum_overall_success_drop_percentage_points"],
            "all_task_success_drops_passed": max(drops.values()) <= rules["maximum_per_task_success_drop_percentage_points"],
        }
        state["comparison"] = {"reference_success_percent": reference["overall"]["pc_success"],
                               "treatment_success_percent": treatment["overall"]["pc_success"],
                               "overall_success_drop_pp": overall_drop, "per_task_success_drop_pp": drops,
                               "median_latency_ms": medians, "latency_reduction_percent": latency_reduction,
                               "criteria": criteria, "engineering_screen_passed": all(criteria.values()),
                               "not_a_statistical_noninferiority_claim": True}
        state.update(stage="completed", completed_at_unix=time.time())
    except Exception as error:
        state.update(stage="failed", error=str(error), failed_at_unix=time.time())
        save()
        raise
    save()
    print(json.dumps(state["comparison"], indent=2), flush=True)


if __name__ == "__main__":
    main()
