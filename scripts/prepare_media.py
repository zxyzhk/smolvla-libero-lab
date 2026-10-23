#!/usr/bin/env python
"""Keep raw videos and export frame-preserving, control-time portfolio copies."""
import argparse
import hashlib
import json
import os
import subprocess
from fractions import Fraction
from pathlib import Path


def probe(path):
    data = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames,duration", "-of", "json", str(path)], text=True))
    return data["streams"][0]


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", default=["phase1_baseline", "phase1_lora", "sampling5_baseline"])
    args = parser.parse_args()
    root = Path(os.environ["PROJECT_ROOT"])
    target = root / "docs"
    instructions = {t["benchmark_task_id"]: t["instruction"] for t in
                    json.loads((root / "configs/data_selection.json").read_text())["selection"]["tasks"]}
    previous_path = root / "reports/media_manifest.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {"videos": []}
    cache = {(v["experiment"], v["task_id"], v["episode_index"]): v for v in previous["videos"]}
    videos = [v for v in previous["videos"] if v["experiment"] not in args.experiments]
    for name in args.experiments:
        evaluation = json.loads((root / f"runs/{name}_eval/eval_info.json").read_text())
        assert evaluation["overall"]["n_episodes"] == 100
        for task in evaluation["per_task"]:
            task_id = task["task_id"]
            assert len(task["metrics"]["video_paths"]) == 20
            for episode_id, source in enumerate(task["metrics"]["video_paths"]):
                source = Path(source)
                source_hash = digest(source)
                key = name, task_id, episode_id
                cached = cache.get(key)
                folder = target / "assets/videos" / name
                folder.mkdir(parents=True, exist_ok=True)
                stem = f"task{task_id}_episode{episode_id:02d}"
                output, poster = folder / f"{stem}.mp4", folder / f"{stem}.jpg"
                original = probe(source)
                reuse = (cached and cached["raw_sha256"] == source_hash and output.is_file()
                         and poster.is_file() and cached.get("display_fps") == 20
                         and cached.get("display_sha256") == digest(output))
                if not reuse:
                    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
                                    "-an", "-vf", "setpts=N/(20*TB)", "-r", "20", "-c:v", "libx264",
                                    "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", "-threads", "2",
                                    "-movflags", "+faststart", str(output)], check=True)
                converted = probe(output)
                if int(original["nb_frames"]) != int(converted["nb_frames"]):
                    raise RuntimeError(f"Conversion changed frame count: {source}: {original} -> {converted}")
                if abs(float(Fraction(converted["r_frame_rate"]))-20) > 1e-8:
                    raise RuntimeError("Incorrect display frame rate")
                poster_s = float(converted["duration"])*.5
                if not reuse:
                    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", str(poster_s),
                                    "-i", str(output), "-frames:v", "1", "-q:v", "3", str(poster)], check=True)
                # Always rebuild labels from the current evaluation, even when reusing media bytes.
                success = task["metrics"]["successes"][episode_id]
                entry = {"experiment": name, "src": str(output.relative_to(target)),
                         "poster": str(poster.relative_to(target)), "task_id": task_id,
                         "episode_index": episode_id, "initial_state_id": episode_id, "seed": 2000+episode_id,
                         "success": success, "instruction": instructions[task_id],
                         "caption": f"任务{task_id} · 初始状态{episode_id} · {'成功' if success else '失败'}。按20Hz仿真控制时间播放，不含推理等待。",
                         "raw_source": str(source), "raw_sha256": source_hash,
                         "display_sha256": digest(output), "frame_count": int(converted["nb_frames"]),
                         "raw_fps": original["r_frame_rate"], "display_fps": 20,
                         "duration_s": float(converted["duration"])}
                videos.append(entry)
                if len(videos) % 20 == 0:
                    print(f"Exported {len(videos)}/300 videos", flush=True)
    report = {"videos": videos, "total_videos": len(videos), "raw_videos_retained": True,
              "display_timebase": "20 Hz simulation control time; model inference waiting excluded",
              "frame_count_preserved": True}
    previous_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({"videos": len(videos), "frame_count_preserved": True}, indent=2))


if __name__ == "__main__":
    main()
