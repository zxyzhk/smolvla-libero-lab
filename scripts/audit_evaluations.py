#!/usr/bin/env python
"""Audit all recorded episode counts, outcomes, initial states and raw videos."""
import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return [100*max(0, center-half), 100*min(1, center+half)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-media", action="store_true", help="Also check all display media hashes and labels")
    args = parser.parse_args()
    root = Path(os.environ["PROJECT_ROOT"])
    results = {}
    raw_paths = set()
    expected_media = {}
    for name, logfile in (("phase1_baseline", "phase1_baseline.log"),
                          ("phase1_lora", "phase1_lora.log"),
                          ("sampling5_baseline", "sampling_ablation.log")):
        path = root / f"runs/{name}_eval/eval_info.json"
        evaluation = json.loads(path.read_text())
        trace = re.findall(r"LIBERO_INIT task_id=(\d+) initial_state_id=(\d+) seed=(\d+)",
                           (root / "logs" / logfile).read_text(errors="replace"))
        assert len(trace) == 100, (name, "initial-state count", len(trace))
        assert [task["task_id"] for task in evaluation["per_task"]] == list(range(5))
        task_results = []
        for task in evaluation["per_task"]:
            task_id = task["task_id"]
            states = [(int(i), int(s)) for t, i, s in trace if int(t) == task_id]
            assert states == [(i, 2000+i) for i in range(20)], (name, task_id, states)
            successes = task["metrics"]["successes"]
            videos = task["metrics"]["video_paths"]
            assert len(successes) == len(videos) == task["n_episodes"] == 20
            assert all(type(value) is bool for value in successes)
            k = sum(successes)
            assert k == task["n_success"] and abs(task["pc_success"] - 100*k/20) < 1e-8
            assert all(abs(a-b) < 1e-6 for a,b in zip(wilson(k,20), task["pc_success_ci95"], strict=True))
            for i, video in enumerate(videos):
                actual = Path(video).resolve()
                expected = (root / f"runs/{name}_eval/videos/libero_spatial_{task_id}/eval_episode_{i}.mp4").resolve()
                assert actual == expected and actual not in raw_paths, (name, task_id, i, actual)
                raw_paths.add(actual)
                assert actual.is_file() and actual.stat().st_size > 0
                expected_media[name, task_id, i] = {"source": actual, "success": successes[i]}
            task_results.append({"task_id": task_id, "initial_state_ids": [i for i,_ in states],
                                 "seeds": [s for _,s in states], "successes": successes,
                                 "n_success": k, "n_episodes": 20, "raw_videos": len(videos)})
        total = sum(t["n_success"] for t in task_results)
        overall = evaluation["overall"]
        assert overall["n_episodes"] == 100 and overall["n_success"] == total
        assert abs(overall["pc_success"]-total) < 1e-8
        assert all(abs(a-b) < 1e-6 for a,b in zip(wilson(total,100), overall["pc_success_ci95"], strict=True))
        results[name] = {"passed": True, "n_success": total, "n_episodes": 100,
                         "eval_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "tasks": task_results}
    assert len(raw_paths) == 300
    if args.require_media:
        media = json.loads((root / "reports/media_manifest.json").read_text())["videos"]
        assert len(media) == 300
        keys, display_paths = set(), set()
        for entry in media:
            key = entry["experiment"], entry["task_id"], entry["episode_index"]
            assert key not in keys and key in expected_media
            keys.add(key)
            expected = expected_media[key]
            assert Path(entry["raw_source"]).resolve() == expected["source"]
            assert entry["success"] is expected["success"]
            assert entry["initial_state_id"] == key[2] and entry["seed"] == 2000+key[2]
            for label, path in (("raw", expected["source"]), ("display", (root / "docs" / entry["src"]).resolve())):
                assert hashlib.file_digest(path.open("rb"), "sha256").hexdigest() == entry[f"{label}_sha256"]
                if label == "display":
                    assert path.is_relative_to((root / "docs/assets/videos").resolve()) and path not in display_paths
                    display_paths.add(path)
            assert (root / "docs" / entry["poster"]).is_file()
            assert entry["display_fps"] == 20 and entry["frame_count"] > 0
        assert keys == set(expected_media)
    report = {"passed": True, "media_passed": args.require_media, "distinct_raw_video_paths": len(raw_paths), "experiments": results,
              "coverage": "3 experiments x 5 tasks x 20 matched initial states; all 300 raw videos present; counts, rates and Wilson intervals recomputed."}
    (root / "reports/evaluation_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: {x:v[x] for x in ("passed", "n_success", "n_episodes")} for k,v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
