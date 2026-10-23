#!/usr/bin/env python
"""Finish an already launched phase: wait, benchmark idle GPUs, and report."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", default="phase1")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--timeout-hours", type=float, default=6)
    args = parser.parse_args()
    root = Path(os.environ["PROJECT_ROOT"])
    destination = root / "reports" / f"{args.phase}_finalize.json"
    result = {"phase": args.phase, "stage": "waiting", "started_at_unix": time.time(),
              "pid": os.getpid(), "errors": []}

    def save():
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(destination)

    save()
    try:
        deadline = time.monotonic() + args.timeout_hours * 3600
        while True:
            states = {}
            for method in ("baseline", "lora"):
                name = f"{args.phase}_{method}"
                try:
                    state = json.loads((root / "reports" / f"{name}_status.json").read_text())
                except (FileNotFoundError, json.JSONDecodeError):
                    continue
                states[method] = state
            result["experiment_stages"] = {method: state["stage"] for method, state in states.items()}
            save()
            if len(states) == 2 and all(s["stage"] in {"completed", "failed"} for s in states.values()):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Finalizer wait expired; existing training processes were not stopped")
            time.sleep(20)

        # Both experiment drivers have exited training/evaluation before timing.
        # Sequential measurements use the same physical GPU to limit device variation.
        result["stage"] = "analysis"
        save()
        for method, state in states.items():
            if state["stage"] != "completed":
                result["errors"].append(f"{method}: {state.get('error', 'experiment failed')}")
                continue
            audit_command = [sys.executable, str(root / "scripts/audit_checkpoint.py"),
                             "--checkpoint", state["checkpoint"], "--name", state["name"]]
            try:
                subprocess.run(audit_command, cwd=root, check=True)
            except subprocess.CalledProcessError as error:
                result["errors"].append(f"{method} checkpoint audit: {error}")
                continue
            if args.skip_latency or state.get("smoke_only"):
                continue
            command = [sys.executable, str(root / "scripts/benchmark_latency.py"),
                       "--checkpoint", state["checkpoint"], "--name", state["name"]]
            print("COMMAND", json.dumps(command), flush=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
            try:
                subprocess.run(command, cwd=root, env=env, check=True)
            except subprocess.CalledProcessError as error:
                result["errors"].append(f"{method} latency: {error}")
        subprocess.run([sys.executable, str(root / "scripts/summarize.py"),
                        "--phase", args.phase], cwd=root, check=True)
        result.update(stage="completed_with_errors" if result["errors"] else "completed",
                      completed_at_unix=time.time())
    except Exception as error:
        result.update(stage="failed", error=str(error), failed_at_unix=time.time())
        save()
        raise
    save()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
