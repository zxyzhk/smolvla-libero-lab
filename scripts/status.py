#!/usr/bin/env python
"""Compact local status; does not start or stop experiments."""
import json
import os
import subprocess
from pathlib import Path

root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
for path in sorted((root / "reports").glob("*_status.json")):
    status = json.loads(path.read_text())
    progress_path = root / "reports" / path.name.replace("_status.json", "_progress.json")
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    print(json.dumps({"name": status["name"], "stage": status["stage"],
                      "gpu": status["physical_gpu"], "step": progress.get("step"),
                      "target_steps": status["steps"],
                      "training_elapsed_s": progress.get("training_elapsed_s"),
                      "peak_allocated_gb": progress.get("peak_allocated_gb"),
                      "error": status.get("error")}, ensure_ascii=False))
subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu,power.draw",
                "--format=csv"], check=False)
