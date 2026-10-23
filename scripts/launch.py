#!/usr/bin/env python
"""Launch one independent experiment per GPU, detached from SSH."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import psutil

parser = argparse.ArgumentParser()
parser.add_argument("--phase", default="phase1")
parser.add_argument("--steps", type=int, default=5000)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--eval-only", action="store_true")
args = parser.parse_args()
root = Path(os.environ["PROJECT_ROOT"])
started = []
for method in ("baseline", "lora"):
    name = f"{args.phase}_{method}"
    record_path = root / "reports" / f"{name}_pid.json"
    if args.eval_only:
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if psutil.pid_exists(record["pid"]):
                process = psutil.Process(record["pid"])
                if process.status() != psutil.STATUS_ZOMBIE and "run_experiment.py" in " ".join(process.cmdline()):
                    raise RuntimeError(f"Experiment still running: {name}")
        if not (root / "runs" / name / "checkpoints/last/pretrained_model").is_dir():
            raise FileNotFoundError(f"No checkpoint for {name}")
    elif (root / "runs" / name).exists() or record_path.exists():
        raise FileExistsError(f"Existing experiment name: {name}")
for method, gpu in [("baseline", 0), ("lora", 1)]:
    name = f"{args.phase}_{method}"
    command = [sys.executable, str(root / "scripts/run_experiment.py"),
               "--method", method, "--gpu", str(gpu), "--phase", args.phase,
               "--steps", str(args.steps), "--batch-size", str(args.batch_size)]
    if args.smoke:
        command.append("--smoke")
    if args.eval_only:
        command.append("--eval-only")
    log_path = root / "logs" / f"{name}.log"
    with log_path.open("a") as log:
        proc = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, env=dict(os.environ))
    record = {"pid": proc.pid, "name": name, "log": str(log_path), "command": command}
    (root / "reports" / f"{name}_pid.json").write_text(json.dumps(record, indent=2))
    started.append(record)
print(json.dumps(started, indent=2))
