#!/usr/bin/env python
"""Initialize LIBERO paths non-interactively and verify an EGL reset/step."""
import argparse
import importlib.util
import hashlib
import json
import os
import tarfile
from pathlib import Path

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configure-only", action="store_true")
    args = parser.parse_args()
    root = Path(os.environ["PROJECT_ROOT"])
    package = Path(importlib.util.find_spec("libero").origin).parent / "libero"
    archive = root / "vendor/libero-master.tar.gz"
    if archive.exists() and not (package / "assets").exists():
        destination = root / "vendor/libero_official"
        destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as handle:
            prefix = handle.getmembers()[0].name.split("/")[0]
            if prefix not in {"LIBERO-master", "LIBERO-8f1084e3132a39270c3a13ebe37270a43ece2a01"}:
                raise ValueError(f"Unexpected official archive root: {prefix}")
            members = [m for m in handle.getmembers()
                       if m.name.startswith(f"{prefix}/libero/libero/assets/")]
            if not members:
                raise ValueError("Archive contains no LIBERO assets")
            handle.extractall(destination, members=members, filter="data")
        assets = destination / prefix / "libero/libero/assets"
        (package / "assets").symlink_to(assets, target_is_directory=True)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        (root / "reports/libero_assets.json").write_text(json.dumps({
            "source_url": "https://codeload.github.com/Lifelong-Robot-Learning/LIBERO/tar.gz/refs/heads/master",
            "archive_sha256": digest, "local_assets": str(assets),
        }, indent=2))
    paths = {"benchmark_root": str(package), "bddl_files": str(package / "bddl_files"),
             "init_states": str(package / "init_files"), "datasets": str(root / "data"),
             "assets": str(package / "assets")}
    config_dir = Path(os.environ["LIBERO_CONFIG_PATH"])
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yaml").write_text(yaml.safe_dump(paths))
    # A prepare_data process started before env.sh was updated may still use this default.
    legacy = Path.home() / ".libero" / "config.yaml"
    if not legacy.exists():
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_text(yaml.safe_dump(paths))
    if args.configure_only:
        print(json.dumps(paths, indent=2))
        return
    from libero.libero import benchmark, get_assets_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    assets = get_assets_path()
    env = OffScreenRenderEnv(
        bddl_file_name=str(package / "bddl_files" / task.problem_folder / task.bddl_file),
        camera_heights=256, camera_widths=256,
    )
    obs = env.reset()
    obs, reward, done, info = env.step(np.zeros(7, dtype=np.float32))
    report = {"task": task.language, "assets": assets,
              "camera_shape": list(obs["agentview_image"].shape),
              "reward": float(reward), "done": bool(done), "egl": True}
    env.close()
    (root / "reports/libero_probe.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
