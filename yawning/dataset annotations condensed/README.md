# YawDD Condensed Frame-Level Annotations

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

- **Total Videos:** 349 (29 Dash, 320 Mirror)
- **Total Frames:** 288,174
- **Total Yawning Frames:** 22,875 (7.94%)
- **Missing Frames Padded:** 791

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
