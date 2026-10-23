#!/usr/bin/env python
"""Official LeRobot trainer with local-only experiment evidence hooks."""
import json
import os
import time
from pathlib import Path

import torch
from lerobot.scripts import lerobot_train as trainer

ROOT = Path(os.environ["PROJECT_ROOT"])
NAME = os.environ.get("EXPERIMENT_NAME", "experiment")
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
captured = {}
original_make_policy = trainer.make_policy
original_processors = trainer.make_pre_post_processors
original_update = trainer.update_policy


def make_policy(*args, **kwargs):
    policy = original_make_policy(*args, **kwargs)
    captured["policy"] = policy
    return policy


def processors(*args, **kwargs):
    policy = captured["policy"]
    parameters = list(policy.named_parameters())
    report = {
        "total_parameters": sum(p.numel() for _, p in parameters),
        "trainable_parameters": sum(p.numel() for _, p in parameters if p.requires_grad),
        "trainable_names": [n for n, p in parameters if p.requires_grad],
        "device": torch.cuda.get_device_name(),
    }
    (REPORTS / f"{NAME}_parameters.json").write_text(json.dumps(report, indent=2))
    print("EXPERIMENT_PARAMETERS", json.dumps({k: v for k, v in report.items() if k != "trainable_names"}), flush=True)
    result = original_processors(*args, **kwargs)
    # LoRA initialization consumes random numbers. Reset after model construction
    # so both recipes start their sampling/noise streams from the same seed.
    torch.manual_seed(1000)
    torch.cuda.manual_seed_all(1000)
    return result


def update(*args, **kwargs):
    if "start" not in captured:
        captured["start"] = time.monotonic()
        captured["step"] = 0
        torch.cuda.reset_peak_memory_stats()
    result = original_update(*args, **kwargs)
    captured["step"] += 1
    step = captured["step"]
    if step == 1 or step % 20 == 0:
        metrics = result[0].to_dict()
        report = {
            "step": step, "training_elapsed_s": time.monotonic() - captured["start"],
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
            "metrics": metrics,
        }
        text = json.dumps(report, default=str)
        with (REPORTS / f"{NAME}_metrics.jsonl").open("a") as handle:
            handle.write(text + "\n")
        (REPORTS / f"{NAME}_progress.json").write_text(text)
    return result


trainer.make_policy = make_policy
trainer.make_pre_post_processors = processors
trainer.update_policy = update

if __name__ == "__main__":
    trainer.main()
