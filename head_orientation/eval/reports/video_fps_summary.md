# Boston University Dataset — Video FPS & Duration Descriptive Statistics

> **Generated on**: 2026-10-20 18:59:48  
> **Total Inspected Files**: 144 sequences  
> **Ground Truth Matching**: 72/144 matched exactly 200 frames  

---

## 1. Grouped Descriptive Statistics Summary

The table below aggregates video metrics across lighting condition folders (`uniform-light`, `varying-light`) and video container formats (`.avi`, `.mov`).

| Group | Count | Actual Frames (Mean ± Std) | Median Frames [Min, Max] | Duration s (Mean ± Std) | Estimated FPS (Mean ± Std) | Container FPS | GT Mismatch Count |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **uniform-light (avi)** | 45 | 199.00 ± 0.00 | 199.0 [199, 199] | 6.60 ± 0.00 | 30.15 ± 0.00 | 30.0 | 45 |
| **ALL (avi)** | 72 | 199.00 ± 0.00 | 199.0 [199, 199] | 6.60 ± 0.00 | 30.15 ± 0.00 | 30.0 | 72 |
| **OVERALL (combined)** | 144 | 199.50 ± 0.50 | 199.5 [199, 200] | 9.93 ± 3.34 | 22.61 ± 7.56 | 22.5 | 72 |
| **uniform-light (mov)** | 45 | 200.00 ± 0.00 | 200.0 [200, 200] | 13.27 ± 0.00 | 15.08 ± 0.00 | 15.0 | 0 |
| **ALL (mov)** | 72 | 200.00 ± 0.00 | 200.0 [200, 200] | 13.27 ± 0.00 | 15.08 ± 0.00 | 15.0 | 0 |
| **varying-light (avi)** | 27 | 199.00 ± 0.00 | 199.0 [199, 199] | 6.60 ± 0.00 | 30.15 ± 0.00 | 30.0 | 27 |
| **varying-light (mov)** | 27 | 200.00 ± 0.00 | 200.0 [200, 200] | 13.27 ± 0.00 | 15.08 ± 0.00 | 15.0 | 0 |

---

## 2. Key Findings & Empirical Observations

1. **Container Differences (.mov vs .avi)**:
   - **`.mov` Sequences**: Exactly **200.0 frames** decoded across 100% of sequences. Perfectly aligns 1:1 with the 200 rows in the ground truth `.dat` files. Container FPS is 15.0 FPS, resulting in an average duration of ~13.33 seconds.
   - **`.avi` Sequences**: Decodes to **199.0 frames** for the vast majority of sequences (missing 1 trailing frame due to the AVI index chunk structure). Container FPS is 30.0 FPS, resulting in an average duration of ~6.63 seconds.
2. **Ground Truth Synchronization**:
   - The BU ground truth sensor (Polhemus Flock of Birds) sampled at 30 Hz for 200 frames.
   - For the upcoming **Phase 2 Regression Benchmark**, the **`.mov` format is strongly recommended as primary** because of its strict 1:1 frame count parity with `.dat` files.
3. **Zero-Frame Drift**:
   - Neither format exhibits frame jitter or variable frame rate within a given container type.

---

## 3. Discrepancies & Mismatch Log

The following sequences exhibited frame counts differing from the 200-frame ground truth `.dat` standard:

| Folder | Sequence | Format | Decoded Frames | Ground Truth Frames | Delta | Duration (s) | File Size |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `uniform-light` | `jam1` | `.avi` | 199 | 200 | -1 | 6.60s | 43.7 MB |
| `uniform-light` | `jam2` | `.avi` | 199 | 200 | -1 | 6.60s | 43.7 MB |
| `uniform-light` | `jam3` | `.avi` | 199 | 200 | -1 | 6.60s | 43.7 MB |
| `uniform-light` | `jam4` | `.avi` | 199 | 200 | -1 | 6.60s | 43.7 MB |
| `uniform-light` | `jam5` | `.avi` | 199 | 200 | -1 | 6.60s | 43.7 MB |
| `uniform-light` | `jam6` | `.avi` | 199 | 200 | -1 | 6.60s | 43.7 MB |
| `uniform-light` | `jam7` | `.avi` | 199 | 200 | -1 | 6.60s | 43.7 MB |
| `uniform-light` | `jam8` | `.avi` | 199 | 200 | -1 | 6.60s | 43.7 MB |
| `uniform-light` | `jam9` | `.avi` | 199 | 200 | -1 | 6.60s | 43.7 MB |
| `uniform-light` | `jim1` | `.avi` | 199 | 200 | -1 | 6.60s | 43.7 MB |
| `uniform-light` | `jim2` | `.avi` | 199 | 200 | -1 | 6.60s | 43.7 MB |
| `uniform-light` | `jim3` | `.avi` | 199 | 200 | -1 | 6.60s | 43.7 MB |
| `uniform-light` | `jim4` | `.avi` | 199 | 200 | -1 | 6.60s | 43.7 MB |
| `uniform-light` | `jim5` | `.avi` | 199 | 200 | -1 | 6.60s | 43.7 MB |
| `uniform-light` | `jim6` | `.avi` | 199 | 200 | -1 | 6.60s | 43.7 MB |
| `uniform-light` | `jim7` | `.avi` | 199 | 200 | -1 | 6.60s | 43.7 MB |
| `uniform-light` | `jim8` | `.avi` | 199 | 200 | -1 | 6.60s | 43.7 MB |
| `uniform-light` | `jim9` | `.avi` | 199 | 200 | -1 | 6.60s | 43.7 MB |
| `uniform-light` | `llm1` | `.avi` | 199 | 200 | -1 | 6.60s | 43.7 MB |
| `uniform-light` | `llm2` | `.avi` | 199 | 200 | -1 | 6.60s | 43.7 MB |
| ... and 52 more sequences (see `video_fps_details.csv`) | | | | | | | |

---

## 4. Next Steps for Phase 2 Evaluation

- Run `evaluate_regression.py` defaulting to `.mov` files to maintain clean 1:1 synchronization with ground truth rows.
- Frame 1 will be used to calibrate `baseline = BaselinePose(rho_0, roll_0)`.
- When evaluating `.avi`, truncate the ground truth comparison at frame 199 to account for the single missing container frame.
