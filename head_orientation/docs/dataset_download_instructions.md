# Boston University (BU) Head Tracking Dataset — Download & Sync Instructions

This guide provides instructions and examples for downloading, auditing, and resuming the [Boston University Head Tracking Dataset](http://csr.bu.edu/headtracking/) using the custom script [`download_bu_dataset.py`](../download-dataset/download_bu_dataset.py).

> [!IMPORTANT]
> **Expected Dataset Location**: The dataset is expected to live inside the **`download-dataset`** folder (`head_orientation/download-dataset/dataset` or `head_orientation/download-dataset/`). All scripts requiring evaluation or test videos look for files in this directory.

---

## 1. Overview of the Dataset

The Boston University Head Tracking Dataset is a standard benchmark used for evaluating head pose estimation algorithms against ground truth measurements collected with a Polhemus Flock of Birds magnetic 3D sensor.

### Remote Directory Structure (`http://csr.bu.edu/headtracking/`)

| Folder | Contents | Description |
|---|---|---|
| `uniform-light/` | 45 sequences (5 subjects × 9 sequences) | Uniform ambient lighting conditions. |
| `varying-light/` | 27 sequences (3 subjects × 9 sequences) | Dynamic illumination variations. |
| `in/` | ~60 test sequences | Specific motion sequences (translations/rotations). |
| `misc/` | Additional test clips | Nodding, shaking gestures. |
| `implementation-data/` | Text files (`eigen*.txt`, `mask.txt`) | PCA eigenvectors and masks from original research. |
| `before_restore/` | **Redundant backup** | Duplicate snapshot from 2009. **Excluded by default**. |

### File Types & Storage Sizes

- **Ground Truth (`.dat`)**: ~12 KB each (**< 1 MB total**). Contains frame-by-frame 3D positions ($T_x, T_y, T_z$) and rotation angles ($R_x, R_y, R_z$).
- **Compressed Video (`.mov`)**: ~2 to 3 MB each (**~200 MB total** across all sequences).
- **Raw Uncompressed Video (`.avi`)**: ~44 MB each (**~5.5 GB total**).

---

## 2. Resuming Interrupted Downloads & Comparison Mode

If a download is interrupted (e.g. terminal closed, script cancelled, connection lost), you **do not** need to re-download everything from scratch:

1. **Automatic HTTP Range Resumption**:
   - The script detects partially downloaded files by comparing local file size against the remote `Content-Length`.
   - It issues an HTTP `Range: bytes=<local_size>-` request, resuming from the exact byte where it left off.
2. **Instant Skipping of Completed Files**:
   - Any file whose local byte count matches the remote server is skipped immediately.
3. **Dataset Audit / Comparison Mode (`--compare`)**:
   - Audits all remote files against your local folder using concurrent worker threads.
   - Outputs a summary of completed files, partially downloaded files, corrupt files, and missing files with progress percentage in seconds.
4. **Auto-Detection of Existing Folder**:
   - If you run the script without specifying `--dest`, it automatically detects your existing dataset directory (e.g. inside `download-dataset`) so you don't accidentally download into the wrong folder.

---

## 3. Command-Line Arguments

[`download_bu_dataset.py`](../download-dataset/download_bu_dataset.py) uses only the Python standard library (`urllib.request`, `html.parser`, `concurrent.futures`, `argparse`). **No external `pip` packages required.**

| Flag | Short | Default | Description |
|---|---|---|---|
| `--dest` | `-d` | Auto-detected / `download-dataset` | Destination folder on your local disk. |
| `--compare`, `--status` | | `False` | Audit remote vs local files and print a report without downloading. |
| `--ext` | | `all` | Filter by file extensions (e.g. `dat`, `mov`, `avi`, `txt`). |
| `--folders` | | `all` | Specific subfolders to crawl (e.g. `uniform-light varying-light`). |
| `--dry-run` | | `False` | Scan remote directory and list matching URLs without downloading. |
| `--force` | | `False` | Force re-download even if file already exists locally. |
| `--include-backup`| | `False` | Include `before_restore/` (duplicate backup). |

---

## 4. Usage Examples

### Example 1: Check Current Progress & Missing Files (`--compare`)
To see which files have finished downloading and which are still missing:
```powershell
# From head_orientation/download-dataset:
python download_bu_dataset.py --compare

# Or from repository root:
python download_bu_dataset.py --compare
```
*Output snippet:*
```text
=================================================================
DATASET COMPARISON REPORT (Remote vs Local)
Target Directory : .../head_orientation/download-dataset/dataset
=================================================================
  [OK]      Complete files      :  242  (5.1 GB)
  [PARTIAL] Partially downloaded:    0  (0.0 B saved, needs resume)
  [ALERT]   Corrupt / Oversized :    0
  [MISSING] Missing files       :   53
-----------------------------------------------------------------
  Overall Progress         : 242/295 files (82.0%)
=================================================================
```

---

### Example 2: Resume Interrupted Download
To resume the download right where it stopped, simply run the script. It will automatically skip completed files and resume any partial files:
```powershell
# Auto-detects the existing download-dataset folder:
python download_bu_dataset.py

# Or explicitly specifying the folder:
python download_bu_dataset.py --dest "head_orientation/download-dataset/dataset"
```

---

### Example 3: Download Only Ground Truth (`.dat` files)
All 72 ground truth annotation files download in ~5 seconds (< 1 MB total):
```powershell
python download_bu_dataset.py --ext dat
```

---

### Example 4: Download Ground Truth + Compact Videos (`.dat` + `.mov`)
If you want video sequences for testing without downloading 5.5 GB of uncompressed AVIs, download `.mov` files (~200 MB total):
```powershell
python download_bu_dataset.py --ext dat mov
```

---

### Example 5: Download Specific Lighting Folders Only
```powershell
python download_bu_dataset.py --folders uniform-light varying-light
```

---

## 5. Understanding Ground Truth `.dat` Files

Each `.dat` file (e.g. `uniform-light/jam1.dat`) corresponds to a video sequence and contains space-separated values:

```
<frame_index> <Tx> <Ty> <Tz> <Rx> <Ry> <Rz>
```

- **`frame_index`**: Frame number (1-indexed).
- **`Tx, Ty, Tz`**: Translation vector in 3D inches measured from the Flock of Birds sensor transmitter.
- **`Rx, Ry, Rz`**: Head rotation Euler angles in degrees:
  - **$R_x$ (Pitch)**: Tilting head up/down.
  - **$R_y$ (Roll)**: Tilting head sideways (ear to shoulder).
  - **$R_z$ (Yaw)**: Turning head left/right.

---

## 6. Directory Layout After Download

```
head_orientation/
├── download-dataset/
│   ├── download_bu_dataset.py
│   └── dataset/
│       ├── uniform-light/
│       │   ├── jam1.avi / jam1.mov / jam1.dat
│       │   └── ...
│       ├── varying-light/
│       │   ├── jal1.avi / jal1.mov / jal1.dat
│       │   └── ...
│       ├── in/
│       ├── misc/
│       └── implementation-data/
└── docs/
    └── dataset_download_instructions.md
```
