#!/usr/bin/env python
"""Verify the saved processor uses this experiment's train-only statistics."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("name must be a filename label")
    root = Path(os.environ["PROJECT_ROOT"])
    checkpoint = args.checkpoint.resolve()
    expected = json.loads((root / "reports/train_only_stats.json").read_text())
    configuration = json.loads((checkpoint / "config.json").read_text())
    checks = []
    for filename, registry in (("policy_preprocessor.json", "normalizer_processor"),
                               ("policy_postprocessor.json", "unnormalizer_processor")):
        processor = json.loads((checkpoint / filename).read_text())
        steps = [step for step in processor["steps"] if step["registry_name"] == registry]
        if len(steps) != 1:
            raise ValueError(f"Expected one {registry} in {filename}")
        step = steps[0]
        buffers = load_file(checkpoint / step["state_file"])
        mapping = step["config"]["norm_map"]
        checks.append({"check": f"{filename}: normalization modes",
                       "passed": mapping["STATE"] == "MEAN_STD" and mapping["ACTION"] == "MEAN_STD"
                       and mapping["VISUAL"] == "IDENTITY"})
        for feature in ("observation.state", "action"):
            for stat in ("mean", "std", "count"):
                value = buffers[f"{feature}.{stat}"].cpu().numpy()
                wanted = np.asarray(expected[feature][stat])
                checks.append({"check": f"{filename}: {feature}.{stat}",
                               "passed": bool(value.shape == wanted.shape
                                              and np.allclose(value, wanted, atol=1e-6, rtol=1e-5)),
                               "max_absolute_error": float(np.max(np.abs(value - wanted)))})
    inputs = configuration["input_features"]
    checks.append({"check": "LIBERO camera, state and action interface",
                   "passed": inputs["observation.state"]["shape"] == [8]
                   and all(inputs[key]["shape"] == [3, 256, 256] for key in
                           ("observation.images.image", "observation.images.image2"))
                   and configuration["output_features"]["action"]["shape"] == [7]})
    report = {"name": args.name, "checkpoint": str(checkpoint), "checks": checks,
              "passed": all(check["passed"] for check in checks),
              "scope": "Saved interfaces and normalization buffers; does not establish policy performance."}
    destination = root / "reports" / f"{args.name}_checkpoint_audit.json"
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("Saved checkpoint processor audit failed")


if __name__ == "__main__":
    main()
