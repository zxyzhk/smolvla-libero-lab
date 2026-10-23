#!/usr/bin/env python
"""Benchmark trained SmolVLA/LoRA checkpoints with explicit BF16 autocast.

Only action-chunk generation is timed. Preprocessing, decoding, simulation,
rendering, and action postprocessing are excluded. Each invocation writes a new
timestamped report; no existing measurements are replaced.
"""
import argparse
import hashlib
import json
import os
import re
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch
from lerobot.configs import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_policy, make_pre_post_processors


def trained_checkpoint(path: Path) -> tuple[Path, dict]:
    """Reject base-model folders and incomplete training checkpoints."""
    checkpoint = path.expanduser().resolve(strict=True)
    if not checkpoint.is_dir():
        raise ValueError(f"Expected a pretrained_model directory: {checkpoint}")
    required = ("config.json", "train_config.json", "policy_preprocessor.json",
                "policy_postprocessor.json")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    step_path = checkpoint.parent / "training_state/training_step.json"
    if missing or not step_path.is_file():
        raise ValueError(
            "Use a complete checkpoint produced by this project's trainer, not a base model. "
            f"Missing files: {missing}; training metadata: {step_path}"
        )
    metadata = json.loads(step_path.read_text())
    if int(metadata.get("step", 0)) <= 0:
        raise ValueError("Checkpoint has no completed training step")
    return checkpoint, metadata


def percentile(values: list[float], quantile: float) -> float:
    """Linearly interpolated sample percentile; preserves raw sample ordering."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measurements", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--sampling-steps", type=int, nargs="+", default=[10, 5])
    parser.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.name):
        parser.error("--name must be a short filename label without directory separators")
    if args.warmup < 1 or args.measurements < 2:
        parser.error("Use at least one warmup and two measurements")
    if any(value <= 0 for value in args.sampling_steps):
        parser.error("Sampling steps must be positive")
    if len(set(args.sampling_steps)) != len(args.sampling_steps):
        parser.error("Sampling step values must be distinct")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support BF16 autocast")

    checkpoint, training_metadata = trained_checkpoint(args.checkpoint)
    root = Path(os.environ["PROJECT_ROOT"])
    manifest_path = root / "configs/data_selection.json"
    manifest_bytes = manifest_path.read_bytes()
    data = json.loads(manifest_bytes)
    heldout = data["selection"]["heldout_episodes"]
    if not heldout:
        raise ValueError("The data manifest contains no held-out episode")
    episode_id = int(heldout[0])
    if episode_id in data["selection"]["train_episodes"]:
        raise ValueError("The benchmark episode also appears in the training split")

    dataset = LeRobotDataset(
        data["dataset"]["repo_id"], root=data["dataset"]["root"],
        revision=data["dataset"]["revision"],
        episodes=[episode_id],
        video_backend=data["dataset"].get("video_backend", "pyav"),
        return_uint8=False,
    )
    cfg = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    if cfg.type != "smolvla":
        raise ValueError(f"This benchmark expects SmolVLA, got {cfg.type!r}")
    adapter = (checkpoint / "adapter_config.json").is_file()
    if bool(cfg.use_peft) != adapter:
        raise ValueError("Checkpoint config and adapter files disagree about use_peft")
    weight_glob = "adapter_model*.safetensors" if adapter else "model*.safetensors"
    if not any(checkpoint.glob(weight_glob)):
        raise FileNotFoundError(f"Missing checkpoint weights: {checkpoint / weight_glob}")
    cfg.pretrained_path = checkpoint
    cfg.device = "cuda"
    # make_policy restores the adapter on its saved base policy when use_peft=True.
    # Processors are loaded from this trained checkpoint below, never rebuilt from
    # the held-out dataset's statistics.
    policy = make_policy(cfg=cfg, ds_meta=dataset.meta)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )

    sample = dataset[0]
    if int(sample["episode_index"].item()) != episode_id:
        raise ValueError("The dataset reader returned an unexpected episode")
    if not isinstance(sample.get("task"), str) or not sample["task"].strip():
        raise ValueError("The held-out observation has no language instruction")
    expected_keys = set(cfg.input_features or {})
    missing_inputs = expected_keys - set(sample)
    if missing_inputs:
        raise ValueError(f"Checkpoint input features absent from dataset: {sorted(missing_inputs)}")
    # Ground-truth actions and episode bookkeeping do not enter the policy input.
    observation = {key: sample[key] for key in expected_keys}
    observation["task"] = sample["task"]
    preprocessor.reset()
    batch = preprocessor(observation)
    if tuple(batch["observation.state"].shape) != (1, 8):
        raise ValueError(f"Unexpected batch state shape: {batch['observation.state'].shape}")

    base = policy.get_base_model() if hasattr(policy, "get_base_model") else policy
    if base.config.adapt_to_pi_aloha:
        raise ValueError("LIBERO benchmark must not enable ALOHA in-place state conversion")
    if base._rtc_enabled():
        raise ValueError("RTC is outside this synchronous action-chunk benchmark")
    device = torch.device("cuda", torch.cuda.current_device())
    original_steps = base.config.num_steps
    parameter_dtypes = Counter(str(parameter.dtype) for parameter in policy.parameters())
    rows = []
    try:
        for sampling_steps in args.sampling_steps:
            base.config.num_steps = sampling_steps
            if base.model.config.num_steps != sampling_steps:
                raise RuntimeError("Sampling-step override did not reach the model")
            base.reset()
            # Repeat the same noise seed for each sampling-step setting. Random
            # initial noise generation remains inside each timed policy call.
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                for _ in range(args.warmup):
                    actions = policy.predict_action_chunk(batch)
                torch.cuda.synchronize(device)
                elapsed = []
                for _ in range(args.measurements):
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    actions = policy.predict_action_chunk(batch)
                    torch.cuda.synchronize(device)
                    elapsed.append((time.perf_counter() - started) * 1000)
            expected_shape = (1, base.config.chunk_size, base.config.action_feature.shape[0])
            if tuple(actions.shape) != expected_shape or not torch.isfinite(actions).all().item():
                raise RuntimeError(f"Invalid generated action chunk: {tuple(actions.shape)}")
            rows.append({
                "num_steps": sampling_steps, "batch_size": 1,
                "chunk_size": base.config.chunk_size,
                "n_action_steps": base.config.n_action_steps,
                "warmup": args.warmup, "measurements": len(elapsed),
                "mean_ms": statistics.mean(elapsed),
                "median_ms": statistics.median(elapsed),
                "p95_ms": percentile(elapsed, 0.95),
                "min_ms": min(elapsed), "max_ms": max(elapsed),
                "action_shape": list(actions.shape), "action_dtype": str(actions.dtype),
                "raw_ms": elapsed,
            })
    finally:
        base.config.num_steps = original_steps

    report = {
        "name": args.name, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint), "checkpoint_step": int(training_metadata["step"]),
        "method": "lora" if adapter else "default_expert_finetune",
        "checkpoint_config_sha256": hashlib.sha256((checkpoint / "config.json").read_bytes()).hexdigest(),
        "gpu": torch.cuda.get_device_name(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "precision": {
            "autocast": "cuda_bfloat16" if args.precision == "bf16" else "cuda_float16",
            "parameter_dtype_counts": dict(parameter_dtypes),
            "note": "Autocast selects compute dtypes; it does not cast all stored parameters.",
        },
        "sample": {
            "dataset_repo_id": data["dataset"]["repo_id"],
            "dataset_revision": data["dataset"]["revision"],
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "episode_index": episode_id, "frame_index": int(sample["frame_index"].item()),
            "task": sample["task"], "heldout": True,
        },
        "seed": args.seed,
        "scope": "Single held-out observation, synchronous model action-chunk generation only; "
                 "excludes preprocessing, decoding, simulation, rendering, postprocessing, and task success.",
        "rows": rows,
    }
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = reports / f"{args.name}_latency_{stamp}_{uuid4().hex[:8]}.json"
    # Exclusive creation makes this safe even if an invocation is repeated.
    with destination.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"REPORT {destination}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
