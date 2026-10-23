#!/usr/bin/env python3
"""Pin, inspect, and select LIBERO trajectories without leaking validation stats.

Run from the project environment after sourcing env.sh. The downloaded Hub tree
is a local experiment copy: only its state/action normalization statistics are
overridden. The original file and the exact episode split are kept in reports/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retry(callable_, attempts: int = 4):
    for attempt in range(attempts):
        try:
            return callable_()
        except Exception:
            if attempt + 1 == attempts:
                raise
            delay = min(2 ** (attempt + 1), 15)
            print(f"Network operation failed; retrying in {delay}s", flush=True)
            time.sleep(delay)


def normalize_instruction(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower().replace("_", " ").strip().rstrip("."))


def repair_episode_tasks(dataset_root: Path, reports_root: Path, revision: str) -> dict:
    """Restore missing per-episode task names from the source frame task indices.

    The pinned video dataset lacks this column, while current LeRobot's
    make_train_eval_datasets uses it to stratify the held-out split by task.
    All original numeric/video references remain unchanged. Only the local copy
    is amended, with original parquet files and both checksums retained.
    """
    episode_files = sorted((dataset_root / "meta/episodes").rglob("*.parquet"))
    if not episode_files:
        raise FileNotFoundError("No episode metadata parquet files found")
    missing_files = [path for path in episode_files if "tasks" not in pq.read_schema(path).names]
    report_path = reports_root / "episode_metadata_repair.json"
    if not missing_files:
        if report_path.exists():
            previous = json.loads(report_path.read_text())
            if previous.get("source_revision") != revision:
                raise RuntimeError("Episode metadata repair belongs to a different source revision")
            return previous
        return {"applied": False, "reason": "Source episode metadata already includes tasks",
                "source_revision": revision}

    tasks_table = pq.read_table(dataset_root / "meta/tasks.parquet")
    task_columns = tasks_table.column_names
    instruction_column = next((name for name in ("__index_level_0__", "task", "tasks", "instruction")
                               if name in task_columns), None)
    if "task_index" not in task_columns or instruction_column is None:
        raise ValueError(f"Unsupported task mapping schema: {task_columns}")
    task_names = {int(row["task_index"]): str(row[instruction_column])
                  for row in tasks_table.to_pylist()}
    episode_task_ids: dict[int, set[int]] = {}
    for data_path in sorted((dataset_root / "data").rglob("*.parquet")):
        table = pq.read_table(data_path, columns=["episode_index", "task_index"])
        episode_ids = table["episode_index"].to_numpy()
        task_ids = table["task_index"].to_numpy()
        # Repeated frame-level pairs are collapsed before accumulating episode labels.
        for episode_id, task_id in np.unique(np.column_stack((episode_ids, task_ids)), axis=0):
            episode_task_ids.setdefault(int(episode_id), set()).add(int(task_id))
    backup_root = reports_root / "source_episode_metadata"
    file_reports = []
    repaired_episodes = 0
    for metadata_path in missing_files:
        relative = metadata_path.relative_to(dataset_root / "meta/episodes")
        backup = backup_root / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(metadata_path, backup)
        table = pq.read_table(metadata_path)
        task_lists = []
        for episode_id in table["episode_index"].to_pylist():
            ids = episode_task_ids.get(int(episode_id), set())
            if not ids or not ids.issubset(task_names):
                raise ValueError(f"Cannot reconstruct tasks for episode {episode_id}: {ids}")
            task_lists.append([task_names[task_id] for task_id in sorted(ids)])
        table = table.append_column("tasks", pa.array(task_lists, type=pa.list_(pa.string())))
        temporary = metadata_path.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary, compression="snappy")
        temporary.replace(metadata_path)
        repaired_episodes += len(task_lists)
        file_reports.append({
            "path": str(metadata_path.relative_to(dataset_root)),
            "backup": str(backup.relative_to(reports_root.parent)),
            "source_sha256": file_hash(backup), "local_sha256": file_hash(metadata_path),
            "episodes": len(task_lists),
        })
    report = {
        "applied": True, "source_revision": revision, "scope": "local dataset copy only",
        "reason": "Source episode metadata is missing tasks required for task-stratified eval_split",
        "method": "Frame episode_index/task_index joined to meta/tasks.parquet task names",
        "repaired_episodes": repaired_episodes, "files": file_reports,
        "upstream_modified": False,
    }
    write_json(report_path, report)
    print(f"Repaired local episode task metadata for {repaired_episodes} episodes", flush=True)
    return report


def vector_stats(values: np.ndarray) -> dict:
    """Exact population moments and quantiles in LeRobot's normalization schema."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"Invalid normalization input: shape={values.shape}")
    result = {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0, ddof=0).tolist(),
        "count": [int(len(values))],
    }
    for quantile in (0.01, 0.10, 0.50, 0.90, 0.99):
        result[f"q{int(quantile * 100):02d}"] = np.quantile(values, quantile, axis=0).tolist()
    return result


def load_numeric_rows(root: Path, episode_ids: list[int]) -> dict[str, np.ndarray]:
    """Read only selected trajectories' numeric columns; no video decoding."""
    columns = ["episode_index", "frame_index", "timestamp", "observation.state", "action"]
    batches = {key: [] for key in columns}
    for parquet_file in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(
            parquet_file,
            columns=columns,
            filters=[("episode_index", "in", episode_ids)],
        )
        if table.num_rows == 0:
            continue
        for key in columns:
            batches[key].append(np.asarray(table[key].to_pylist()))
    if not batches["episode_index"]:
        raise RuntimeError("No data rows matched the selected episodes")
    return {key: np.concatenate(parts, axis=0) for key, parts in batches.items()}


def camera_preview(dataset_root: Path, repo_id: str, revision: str, episode_id: int,
                   reports_root: Path) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image

    dataset = LeRobotDataset(
        repo_id=repo_id, root=dataset_root, revision=revision,
        episodes=[episode_id], video_backend="pyav", return_uint8=True,
    )
    sample = dataset[0]
    camera_report = {}
    previews = []
    for key in dataset.meta.camera_keys:
        tensor = sample[key]
        array = tensor.detach().cpu().numpy()
        camera_report[key] = {
            "shape": list(array.shape), "dtype": str(array.dtype),
            "min": float(array.min()), "max": float(array.max()),
        }
        if array.ndim != 3:
            raise ValueError(f"Unexpected camera frame shape for {key}: {array.shape}")
        if array.shape[0] == 3:
            array = array.transpose(1, 2, 0)
        if array.dtype != np.uint8:
            array = np.clip(array * 255 if array.max() <= 1 else array, 0, 255).astype(np.uint8)
        previews.append(Image.fromarray(array))
    if previews:
        canvas = Image.new("RGB", (sum(im.width for im in previews), max(im.height for im in previews)))
        x = 0
        for preview in previews:
            canvas.paste(preview, (x, 0))
            x += preview.width
        canvas.save(reports_root / "sample_cameras.png")
    return {
        "episode_index": episode_id,
        "task": str(sample.get("task", "")),
        "state_shape": list(sample["observation.state"].shape),
        "action_shape": list(sample["action"].shape),
        "cameras": camera_report,
        "preview": "reports/sample_cameras.png",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path,
                        default=Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1])))
    parser.add_argument("--repo-id", default="lerobot/libero")
    parser.add_argument("--revision", default="a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4")
    parser.add_argument("--task-ids", default="0,1,2,3,4")
    parser.add_argument("--train-per-task", type=int, default=20)
    parser.add_argument("--heldout-per-task", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    if args.train_per_task <= 0 or args.heldout_per_task <= 0:
        raise ValueError("Both train and held-out episode counts must be positive")
    task_ids = [int(value) for value in args.task_ids.split(",")]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Duplicate benchmark task ids")
    project_root = args.project_root.resolve()
    dataset_root = project_root / "data/libero"
    reports_root = project_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    api = HfApi(endpoint=endpoint, token=False)
    source_info = retry(lambda: api.dataset_info(args.repo_id, revision=args.revision, timeout=30))
    revision = source_info.sha
    if not revision:
        raise RuntimeError("Hub did not return a dataset commit SHA")
    print(f"Dataset: {args.repo_id}@{revision}", flush=True)
    previous_manifest_path = project_root / "configs/data_selection.json"
    if previous_manifest_path.exists():
        previous = json.loads(previous_manifest_path.read_text())
        if previous["dataset"]["revision"] != revision:
            raise RuntimeError("This local tree belongs to a different dataset revision; use a new project root")
    if not args.skip_download:
        retry(lambda: snapshot_download(
            repo_id=args.repo_id, repo_type="dataset", revision=revision,
            local_dir=dataset_root, endpoint=endpoint, token=False,
            allow_patterns=["meta/**", "data/**", "videos/**", "README.md"],
            max_workers=args.max_workers,
        ))

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from libero.libero import benchmark

    metadata_repair = repair_episode_tasks(dataset_root, reports_root, revision)
    metadata = LeRobotDatasetMetadata(args.repo_id, root=dataset_root, revision=revision, token=False)
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    episode_rows = list(metadata.episodes)
    if not all(int(row.get("episode_index", index)) == index for index, row in enumerate(episode_rows)):
        raise ValueError("Episode metadata is not in absolute episode-index order")
    task_to_episodes = {}
    for index, row in enumerate(episode_rows):
        for instruction in row.get("tasks", []):
            task_to_episodes.setdefault(normalize_instruction(str(instruction)), []).append(index)

    selected_episodes, train_episodes, heldout_episodes, selections = [], [], [], []
    for task_id in task_ids:
        task = suite.get_task(task_id)
        instruction = task.language
        candidates = sorted(set(task_to_episodes.get(normalize_instruction(instruction), [])))
        required = args.train_per_task + args.heldout_per_task
        if len(candidates) < required:
            available = sorted(task_to_episodes)
            raise ValueError(
                f"Task {task_id}: {instruction!r} has {len(candidates)} matching episodes, "
                f"requires {required}. Available instructions: {available}"
            )
        # Per-task RNG keeps the split stable if the list of other selected tasks changes.
        shuffled = candidates.copy()
        random.Random(args.seed + task_id).shuffle(shuffled)
        train = shuffled[:args.train_per_task]
        heldout = shuffled[args.train_per_task:required]
        selected_episodes.extend(train + heldout)
        train_episodes.extend(train)
        heldout_episodes.extend(heldout)
        selections.append({
            "benchmark_task_id": task_id, "benchmark_task_name": task.name,
            "instruction": instruction, "available_episodes": len(candidates),
            "train_episodes": train, "heldout_episodes": heldout,
        })
    if set(train_episodes) & set(heldout_episodes):
        raise AssertionError("Training and held-out episodes overlap")
    print(f"Selected {len(train_episodes)} training and {len(heldout_episodes)} held-out episodes", flush=True)
    numeric = load_numeric_rows(dataset_root, selected_episodes)
    train_mask = np.isin(numeric["episode_index"], train_episodes)
    heldout_mask = np.isin(numeric["episode_index"], heldout_episodes)
    state = numeric["observation.state"]
    action = numeric["action"]
    if state.ndim != 2 or state.shape[1] != 8 or action.ndim != 2 or action.shape[1] != 7:
        raise ValueError(f"Expected LIBERO state/action widths 8/7; got {state.shape}/{action.shape}")
    for key in ("observation.state", "action", "timestamp"):
        if not np.isfinite(numeric[key]).all():
            raise ValueError(f"Non-finite values in selected data column {key}")
    observed = set(numeric["episode_index"].astype(int).tolist())
    if observed != set(selected_episodes):
        raise ValueError("Some selected episode ids have no data rows")
    temporal_checks = []
    for episode_id in selected_episodes:
        mask = numeric["episode_index"] == episode_id
        order = np.argsort(numeric["frame_index"][mask])
        indices = numeric["frame_index"][mask][order]
        timestamps = numeric["timestamp"][mask][order]
        valid_frames = bool(np.all(np.diff(indices) == 1))
        valid_times = bool(np.all(np.diff(timestamps) > 0))
        if not valid_frames or not valid_times:
            raise ValueError(f"Non-contiguous frames or timestamps in episode {episode_id}")
        temporal_checks.append({"episode_index": episode_id, "frames": int(len(indices)),
                                "duration_s": float(timestamps[-1] - timestamps[0])})

    stats_path = dataset_root / "meta/stats.json"
    source_stats_backup = reports_root / "source_stats.json"
    if not source_stats_backup.exists():
        shutil.copy2(stats_path, source_stats_backup)
    stats = json.loads(source_stats_backup.read_text())
    training_stats = {
        "observation.state": vector_stats(state[train_mask]),
        "action": vector_stats(action[train_mask]),
    }
    stats.update(training_stats)
    write_json(reports_root / "train_only_stats.json", training_stats)
    write_json(stats_path, stats)
    normalization = {
        "scope": "training trajectories only; held-out trajectories excluded",
        "overridden_keys": ["observation.state", "action"],
        "population_std_ddof": 0,
        "training_frames": int(train_mask.sum()),
        "source_stats_sha256": file_hash(source_stats_backup),
        "local_stats_sha256": file_hash(stats_path),
        "source_stats_backup": "reports/source_stats.json",
        "note": "Image stats are preserved; SmolVLA uses VISUAL=IDENTITY. No Hub writes are performed.",
    }
    sample = camera_preview(dataset_root, args.repo_id, revision, train_episodes[0], reports_root)
    manifest = {
        "schema_version": 1,
        "dataset": {
            "repo_id": args.repo_id, "root": str(dataset_root), "revision": revision,
            "episodes": selected_episodes,
            "eval_split": args.heldout_per_task / (args.train_per_task + args.heldout_per_task),
            "video_backend": "pyav", "use_imagenet_stats": False,
        },
        "benchmark": {"suite": "libero_spatial", "task_ids": task_ids},
        "selection": {"seed": args.seed, "train_per_task": args.train_per_task,
                      "heldout_per_task": args.heldout_per_task, "tasks": selections,
                      "train_episodes": train_episodes, "heldout_episodes": heldout_episodes},
        "normalization": normalization,
        "local_metadata_repair": metadata_repair,
        "source_endpoint": endpoint,
    }
    write_json(project_root / "configs/data_selection.json", manifest)
    audit = {
        "source": {"repo_id": args.repo_id, "revision": revision, "endpoint": endpoint},
        "benchmark": manifest["benchmark"], "selection": manifest["selection"],
        "normalization": normalization,
        "local_metadata_repair": metadata_repair,
        "dataset_total_episodes": int(metadata.total_episodes),
        "dataset_total_frames": int(metadata.total_frames),
        "selected_frames": int(len(state)), "training_frames": int(train_mask.sum()),
        "heldout_frames": int(heldout_mask.sum()),
        "state_dimensions": int(state.shape[1]), "action_dimensions": int(action.shape[1]),
        "finite_state_action_timestamp": True, "trajectory_checks": temporal_checks,
        "training_distribution": training_stats,
        "zero_std_dimensions": {key: np.where(np.asarray(value["std"]) == 0)[0].tolist()
                                for key, value in training_stats.items()},
        "sample": sample,
        "evaluation_scope": "Five selected LIBERO-Spatial tasks; not the full LIBERO benchmark",
    }
    write_json(reports_root / "data_audit.json", audit)
    print(json.dumps({"manifest": str(project_root / "configs/data_selection.json"),
                      "audit": str(reports_root / "data_audit.json"),
                      "train_frames": int(train_mask.sum()), "heldout_frames": int(heldout_mask.sum()),
                      "sample": sample}, indent=2), flush=True)


if __name__ == "__main__":
    main()
