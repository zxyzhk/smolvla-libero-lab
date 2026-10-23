#!/usr/bin/env python
"""Apply recorded compatibility patches to the pinned upstream checkout."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
repo = root / "vendor/lerobot"
records = []
for path in sorted((root / "patches").glob("*.patch")):
    check = subprocess.run(["git", "apply", "--check", str(path)], cwd=repo, capture_output=True)
    if check.returncode == 0:
        subprocess.run(["git", "apply", str(path)], cwd=repo, check=True)
    else:
        subprocess.run(["git", "apply", "--reverse", "--check", str(path)], cwd=repo, check=True)
    records.append({"path": str(path.relative_to(root)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
report = {"upstream_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
          "patches": records,
          "purpose": "Record evaluation compatibility fixes: freeze completed sync lanes, trace initial states, and retain all complete episode videos."}
(root / "reports/upstream_patches.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
