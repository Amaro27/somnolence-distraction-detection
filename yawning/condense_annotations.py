import os
import time
import json
import sys

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def format_time(seconds):
    seconds = int(seconds)
    mins, secs = divmod(seconds, 60)
    if mins > 0:
        return f"{mins}m {secs:02d}s"
    return f"{secs}s"


def render_progress_bar(current, total, prefix="", suffix="", bar_len=25):
    fraction = current / max(1, total)
    filled_len = int(bar_len * fraction)
    bar = "=" * filled_len + (">" if filled_len < bar_len else "") + "." * (bar_len - filled_len - 1 if filled_len < bar_len else 0)
    percent = fraction * 100.0
    line = f"\r{prefix} [{bar}] {current}/{total} ({percent:5.1f}%) {suffix}"
    # Pad with spaces to clear any previous longer line in terminal
    line = line.ljust(95)
    sys.stdout.write(line)
    sys.stdout.flush()


def load_or_extract_video_frame_counts(videos_base, script_dir):
    cache_path = os.path.join(script_dir, "video_frame_counts.json")
    if os.path.isfile(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    counts = {}
    if not HAS_CV2 or not os.path.isdir(videos_base):
        return counts

    for root, _, files in os.walk(videos_base):
        for f in files:
            if f.lower().endswith(".avi"):
                clean = f.replace(".avi.avi", "").replace(".avi", "").strip()
                vpath = os.path.join(root, f)
                cap = cv2.VideoCapture(vpath)
                if cap.isOpened():
                    counts[clean] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)

    return counts


def condense_dataset(force_recompute=False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_base = os.path.join(script_dir, "dataset annotations")
    videos_base = os.path.join(script_dir, "YawDD dataset")
    output_base = os.path.join(script_dir, "dataset annotations condensed")

    print("=" * 70, flush=True)
    print("YawDD Dataset Annotation Condensation (Real-Time Progress)", flush=True)
    print(f"Input:   {input_base}", flush=True)
    print(f"Output:  {output_base}", flush=True)
    print("=" * 70, flush=True)

    t_start = time.time()

    # Step 1: Video frame counts
    print("[1/4] Loading video frame counts...", flush=True)
    video_counts = load_or_extract_video_frame_counts(videos_base, script_dir)
    print(f"      Mapped frame counts for {len(video_counts)} videos.", flush=True)

    cameras = ["Dash", "Mirror"]

    master_data = {
        "metadata": {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Condensed frame-level yawn annotations for YawDD (1 = Yawn, 0 = No-Yawn)",
            "encoding": "single continuous binary string per video where index i matches video frame i",
            "camera_counts": {}
        },
        "annotations": {
            "Dash": {},
            "Mirror": {}
        }
    }

    total_videos = 0
    total_frames = 0
    total_yawn_frames = 0
    missing_frame_count = 0
    resumed_count = 0

    print("[2/4] Processing directories and condensing labels...", flush=True)
    for cam_name in cameras:
        cam_in_dir = os.path.join(input_base, cam_name)
        cam_out_dir = os.path.join(output_base, cam_name)
        os.makedirs(cam_out_dir, exist_ok=True)

        if not os.path.isdir(cam_in_dir):
            print(f"Warning: {cam_in_dir} not found.", flush=True)
            continue

        video_dirs = sorted([d for d in os.listdir(cam_in_dir) if os.path.isdir(os.path.join(cam_in_dir, d))])
        total_vids_cam = len(video_dirs)
        print(f"\n---> Camera: {cam_name} ({total_vids_cam} videos)", flush=True)

        t_cam_start = time.time()
        for idx, v_name in enumerate(video_dirs, 1):
            out_txt_path = os.path.join(cam_out_dir, f"{v_name}.txt")

            # Check if already processed (Resume capability)
            if not force_recompute and os.path.isfile(out_txt_path) and os.path.getsize(out_txt_path) > 0:
                with open(out_txt_path, "r", encoding="utf-8") as rf:
                    seq_str = rf.read().strip()
                v_fc = len(seq_str)
                v_yawns = seq_str.count("1")

                master_data["annotations"][cam_name][v_name] = {
                    "frame_count": v_fc,
                    "yawn_frames": v_yawns,
                    "labels": seq_str
                }
                total_videos += 1
                total_frames += v_fc
                total_yawn_frames += v_yawns
                resumed_count += 1

                render_progress_bar(
                    idx, total_vids_cam,
                    prefix=f"[{cam_name}]",
                    suffix=f"[Cached] {v_name[:20]}"
                )
                continue

            labels_dir = os.path.join(cam_in_dir, v_name, "labels")
            if not os.path.isdir(labels_dir):
                continue

            frame_map = {}
            for entry in os.scandir(labels_dir):
                if entry.name.endswith(".txt"):
                    parts = entry.name[:-4].split("_")
                    try:
                        f_idx = int(parts[0])
                    except ValueError:
                        continue

                    with open(entry.path, "rb") as fp:
                        val = 1 if fp.read(1) == b"1" else 0

                    if val == 1 or f_idx not in frame_map:
                        frame_map[f_idx] = val

            max_label_idx = max(frame_map.keys()) if frame_map else -1
            v_fc = video_counts.get(v_name, -1)
            target_fc = max(v_fc, max_label_idx + 1)

            chars = []
            v_yawns = 0
            for i in range(target_fc):
                val = frame_map.get(i, 0)
                if i not in frame_map:
                    missing_frame_count += 1
                if val == 1:
                    v_yawns += 1
                    chars.append("1")
                else:
                    chars.append("0")

            seq_str = "".join(chars)

            with open(out_txt_path, "w", encoding="utf-8") as out_fp:
                out_fp.write(seq_str)

            master_data["annotations"][cam_name][v_name] = {
                "frame_count": target_fc,
                "yawn_frames": v_yawns,
                "labels": seq_str
            }

            total_videos += 1
            total_frames += target_fc
            total_yawn_frames += v_yawns

            # Calculate ETA
            elapsed_cam = time.time() - t_cam_start
            vids_done_in_run = idx - (resumed_count if idx >= resumed_count else 0)
            if vids_done_in_run > 0:
                avg_time_per_vid = elapsed_cam / vids_done_in_run
                vids_left = total_vids_cam - idx
                eta_sec = avg_time_per_vid * vids_left
                eta_str = f"ETA: {format_time(eta_sec)}"
            else:
                eta_str = ""

            render_progress_bar(
                idx, total_vids_cam,
                prefix=f"[{cam_name}]",
                suffix=f"{v_name[:20]} ({target_fc}f) | {eta_str}"
            )

        print("", flush=True)  # New line after camera completes
        print(f"      Completed {cam_name} in {format_time(time.time() - t_cam_start)}.", flush=True)
        master_data["metadata"]["camera_counts"][cam_name] = len(video_dirs)

    # Step 3: Write master annotations.json
    print("\n[3/4] Writing master annotations.json...", flush=True)
    master_json_path = os.path.join(output_base, "annotations.json")
    with open(master_json_path, "w", encoding="utf-8") as jf:
        json.dump(master_data, jf, indent=2)

    # Step 4: Write README.md in condensed folder
    print("[4/4] Writing README.md...", flush=True)
    readme_path = os.path.join(output_base, "README.md")
    readme_content = f"""# YawDD Condensed Frame-Level Annotations

This directory contains condensed ground-truth yawn annotations for the **YawDD** dataset, compressed from 287,000+ individual per-frame text files down into **349 single-line text files** plus a master JSON file.

## Dataset Structure

```
dataset annotations condensed/
├── Dash/
│   ├── 1-FemaleNoGlasses.txt
│   ├── ... (29 videos)
├── Mirror/
│   ├── 1-FemaleNoGlasses-Normal.txt
│   ├── 1-FemaleNoGlasses-Yawning.txt
│   ├── ... (320 videos)
├── annotations.json      # Master file with all 349 videos and frame strings
└── README.md
```

## Format Specification

- Each `.txt` file contains a **single continuous string of `0` and `1` characters** (e.g., `0000000000111111111000000000...`).
- **`0`** = No-Yawn (or normal/talking/unannotated)
- **`1`** = Yawn
- **1:1 Alignment**: Character at index `i` corresponds **exactly to video frame `i`** (0-indexed).
- Any unannotated frames from the original pipeline were padded with `0` so length matches the video duration.

## Statistics

- **Total Videos:** {total_videos} (29 Dash, 320 Mirror)
- **Total Frames:** {total_frames:,}
- **Total Yawning Frames:** {total_yawn_frames:,} ({total_yawn_frames / total_frames * 100:.2f}%)
- **Missing Frames Padded:** {missing_frame_count:,}

## Quick Usage in Python

### Reading a single video's labels:
```python
# Option A: Direct string indexing
with open("yawning/dataset annotations condensed/Mirror/1-FemaleNoGlasses-Yawning.txt") as f:
    labels = f.read().strip()
is_yawn = (labels[frame_idx] == '1')

# Option B: As a Python list of ints
with open("yawning/dataset annotations condensed/Mirror/1-FemaleNoGlasses-Yawning.txt") as f:
    labels = [int(c) for c in f.read().strip()]

# Option C: As a fast NumPy array
import numpy as np
with open("yawning/dataset annotations condensed/Mirror/1-FemaleNoGlasses-Yawning.txt") as f:
    labels = np.frombuffer(f.read().strip().encode('ascii'), dtype='u1') - 48
```

### Reading the entire dataset in 1 call:
```python
import json

with open("yawning/dataset annotations condensed/annotations.json") as f:
    dataset = json.load(f)

# Access any video instantly:
dash_labels = dataset["annotations"]["Dash"]["1-FemaleNoGlasses"]["labels"]
mirror_labels = dataset["annotations"]["Mirror"]["1-FemaleNoGlasses-Yawning"]["labels"]
```
"""
    with open(readme_path, "w", encoding="utf-8") as rf:
        rf.write(readme_content)

    t_end = time.time()
    elapsed = t_end - t_start

    print("\n" + "=" * 70, flush=True)
    print("Condensation Completed Successfully!", flush=True)
    print(f"Total Videos:          {total_videos} (Resumed: {resumed_count}, Newly Processed: {total_videos - resumed_count})", flush=True)
    print(f"Total Frames:          {total_frames:,}", flush=True)
    print(f"Total Yawn Frames:     {total_yawn_frames:,} ({total_yawn_frames / total_frames * 100:.2f}%)", flush=True)
    print(f"Missing Frames Padded: {missing_frame_count:,}", flush=True)
    print(f"Total Elapsed Time:    {format_time(elapsed)}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    force = "--force" in sys.argv
    condense_dataset(force_recompute=force)
