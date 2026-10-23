#!/usr/bin/env python
"""Download public, revision-pinned model artifacts; never upload anything."""
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main():
    root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
    api = HfApi()
    manifest = {"endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"), "models": {}}
    repos = {
        "smolvla_base": ("lerobot/smolvla_base", "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6"),
        "smolvlm": ("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "7b375e1b73b11138ff12fe22c8f2822d8fe03467"),
    }
    for name, (repo, requested_revision) in repos.items():
        revision = api.model_info(repo, revision=requested_revision, token=False).sha
        destination = root / "models" / name
        snapshot_download(
            repo_id=repo, revision=revision, local_dir=destination,
            ignore_patterns=["*.png", "*.jpg", "*.mp4", "*.gif", "*.bin", "*.onnx"],
            max_workers=4, token=False,
        )
        weights = {}
        for path in destination.glob("*.safetensors"):
            h = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    h.update(chunk)
            weights[path.name] = {"size": path.stat().st_size, "sha256": h.hexdigest()}
        if "model.safetensors" not in weights:
            raise FileNotFoundError(f"Missing model weights in {destination}")
        manifest["models"][name] = {
            "repo_id": repo, "revision": revision, "local_path": str(destination), "weights": weights,
        }
        print(f"Prepared {repo} at {revision}", flush=True)

    # Local configuration overrides only: retain the original files as provenance.
    base = root / "models" / "smolvla_base"
    vlm_path = str(root / "models" / "smolvlm")
    overrides = []

    def replace_remote_paths(value):
        if isinstance(value, dict):
            return {k: (vlm_path if k in {"vlm_model_name", "tokenizer_name"} and isinstance(v, str)
                        and v.startswith("HuggingFaceTB/SmolVLM2") else replace_remote_paths(v))
                    for k, v in value.items()}
        if isinstance(value, list):
            return [replace_remote_paths(v) for v in value]
        return value

    for path in base.glob("*.json"):
        if path.name.endswith(".original.json"):
            continue
        original = json.loads(path.read_text())
        patched = replace_remote_paths(original)
        if original != patched:
            backup = path.with_suffix(".original.json")
            if not backup.exists():
                backup.write_text(json.dumps(original, indent=2))
            path.write_text(json.dumps(patched, indent=2))
            overrides.append(path.name)
    manifest["local_config_overrides"] = overrides
    (root / "reports").mkdir(exist_ok=True)
    (root / "reports" / "models.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
