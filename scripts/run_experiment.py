#!/usr/bin/env python
"""Run one reproducible training + closed-loop evaluation experiment on one GPU."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["baseline", "lora"], required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--phase", default="phase1")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--eval-only", action="store_true", help="Retry evaluation from an existing checkpoint")
    args = parser.parse_args()
    root = Path(os.environ["PROJECT_ROOT"])
    manifest = json.loads((root / "configs/data_selection.json").read_text())
    dataset = manifest["dataset"]
    name = f"{args.phase}_{args.method}"
    output = root / "runs" / name
    if output.exists() and not args.eval_only:
        raise FileExistsError(f"Refusing to overwrite existing run: {output}")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
               MUJOCO_EGL_DEVICE_ID=str(args.gpu), EXPERIMENT_NAME=name)
    state_path = root / "reports" / f"{name}_status.json"
    state = {"name": name, "method": args.method, "physical_gpu": args.gpu,
             "steps": args.steps, "batch_size": args.batch_size,
             "seed": 1000, "output_dir": str(output), "started_at_unix": time.time(),
             "smoke_only": args.smoke, "stage": "training"}
    if args.eval_only:
        state = json.loads(state_path.read_text())
        if state["smoke_only"] != args.smoke or state["physical_gpu"] != args.gpu:
            raise ValueError("Evaluation retry must preserve smoke mode and physical GPU")
        if state["stage"] == "completed":
            raise ValueError("Evaluation already completed; use a new experiment for another comparison")
        state.setdefault("evaluation_retries", []).append({
            "previous_stage": state["stage"], "previous_error": state.get("error"),
            "retry_at_unix": time.time(),
        })
        state.pop("error", None)

    def save():
        state_path.write_text(json.dumps(state, indent=2))

    def run(command):
        print("COMMAND", json.dumps(command), flush=True)
        started = time.monotonic()
        subprocess.run(command, cwd=root, env=env, check=True)
        return time.monotonic() - started

    command = [sys.executable, str(root / "scripts/train_entry.py"),
        f"--policy.path={root}/models/smolvla_base",
        "--policy.input_features=null", "--policy.output_features=null",
        "--policy.device=cuda", "--policy.push_to_hub=false",
        "--policy.load_vlm_weights=true", "--policy.freeze_vision_encoder=true",
        "--policy.train_expert_only=true", "--policy.train_state_proj=true",
        "--policy.chunk_size=50", "--policy.n_action_steps=10", "--policy.num_steps=10",
        f"--policy.vlm_model_name={root}/models/smolvlm",
        f"--dataset.repo_id={dataset['repo_id']}", f"--dataset.root={dataset['root']}",
        f"--dataset.revision={dataset['revision']}",
        f"--dataset.episodes={json.dumps(dataset['episodes'])}",
        f"--dataset.eval_split={dataset['eval_split']}",
        f"--dataset.video_backend={dataset.get('video_backend', 'pyav')}",
        "--dataset.use_imagenet_stats=false", "--dataset.return_uint8=true",
        "--env.type=libero", "--env.task=libero_spatial", "--env.task_ids=[0,1,2,3,4]",
        "--env.observation_height=256", "--env.observation_width=256",
        "--env.max_parallel_tasks=1", "--env_eval_freq=0",
        f"--steps={args.steps}", f"--batch_size={args.batch_size}", "--seed=1000",
        "--num_workers=4", "--prefetch_factor=2", "--log_freq=20",
        f"--save_freq={args.steps if args.smoke else max(1, args.steps // 2)}",
        f"--eval_steps={0 if args.smoke else 1000}", "--max_eval_samples=160",
        "--accelerator.mixed_precision=bf16", "--wandb.enable=false",
        f"--output_dir={output}", f"--job_name={name}"]
    if args.method == "lora":
        command += ["--peft.method_type=LORA", "--peft.r=32", "--peft.lora_alpha=32",
                    "--policy.optimizer_lr=1e-3", "--policy.scheduler_decay_lr=1e-4"]
    else:
        command += ["--policy.optimizer_lr=1e-4", "--policy.scheduler_decay_lr=2.5e-6"]
    command += [f"--policy.scheduler_warmup_steps={min(200, max(1, args.steps // 10))}",
                f"--policy.scheduler_decay_steps={args.steps}"]
    try:
        if not args.eval_only:
            state["train_command"] = command
            save()
            state["training_wall_s"] = run(command)
        checkpoint = output / "checkpoints/last/pretrained_model"
        if not checkpoint.is_dir():
            raise FileNotFoundError(checkpoint)
        state.update(stage="evaluating", checkpoint=str(checkpoint))
        state.setdefault("trained_at_unix", time.time())
        save()
        eval_dir = root / "runs" / f"{name}_eval"
        command = [sys.executable, "-m", "lerobot.scripts.lerobot_eval",
            f"--policy.path={checkpoint}", "--policy.device=cuda", "--policy.use_amp=true",
            "--policy.num_steps=10", "--policy.n_action_steps=10",
            "--env.type=libero", "--env.task=libero_spatial",
            f"--env.task_ids={'[0]' if args.smoke else '[0,1,2,3,4]'}",
            "--env.observation_height=256", "--env.observation_width=256",
            "--env.init_states=true", "--env.hard_reset=true", "--env.max_parallel_tasks=1",
            f"--eval.n_episodes={1 if args.smoke else 20}",
            f"--eval.batch_size={1 if args.smoke else 4}", "--eval.use_async_envs=false",
            "--seed=2000", f"--output_dir={eval_dir}"]
        if args.smoke:
            command += ["--env.episode_length=40"]
        state["eval_command"] = command
        save()
        state["evaluation_wall_s"] = run(command)
        state.update(stage="completed", completed_at_unix=time.time(), eval_dir=str(eval_dir))
    except Exception as error:
        state.update(stage="failed", failed_at_unix=time.time(), error=str(error))
        save()
        raise
    save()


if __name__ == "__main__":
    main()
