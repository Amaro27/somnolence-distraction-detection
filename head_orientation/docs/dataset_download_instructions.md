# Boston University (BU) Head Tracking Dataset — Download Instructions

This guide provides instructions and examples for downloading the [Boston University Head Tracking Dataset](http://csr.bu.edu/headtracking/) using the custom script [`download_bu_dataset.py`](../download_bu_dataset.py).

---

## 1. Overview of the Dataset

The Boston University Head Tracking Dataset is a standard benchmark used for evaluating head pose estimation algorithms against ground truth measurements collected with a Polhemus Flock of Birds magnetic 3D sensor.

### Remote Directory Structure (`http://csr.bu.edu/headtracking/`)

| Folder | Contents | Description |
|---|---|---|
| `uniform-light/` | 45 sequences (5 subjects × 9 sequences) | Uniform ambient lighting conditions. |
| `varying-light/` | 27 sequences (3 subjects × 9 sequences) | Lighting conditions changing dynamically. |
| `in/` | ~60 test sequences | Specific motion sequences (translations/rotations). |
| `misc/` | Additional test clips | Nodding, shaking gestures. |
| `implementation-data/` | Text files (`eigen*.txt`, `mask.txt`) | PCA eigenvectors and masks from original research. |
| `before_restore/` | **Redundant backup** | Duplicate snapshot from 2009. **Excluded by default**. |

### File Types & Storage Sizes

- **Ground Truth (`.dat`)**: ~12 KB each (**< 1 MB total**). Contains frame-by-frame 3D positions ($T_x, T_y, T_z$) and rotation angles ($R_x, R_y, R_z$).
- **Compressed Video (`.mov`)**: ~2 to 3 MB each (**~200 MB total** across all sequences).
- **Raw Uncompressed Video (`.avi`)**: ~44 MB each (**~5.5 GB total**).

---

## 2. Script Requirements & Options

[`download_bu_dataset.py`](../download_bu_dataset.py) is built purely with Python's standard library (`urllib.request`, `html.parser`, `argparse`). **No external `pip` packages are needed.**

### Command-Line Arguments

| Flag | Short | Default | Description |
|---|---|---|---|
| `--dest` | `-d` | `data/bu_headtracking` | Destination directory on your system. |
| `--ext` | | `all` | Filter by file extensions (e.g. `dat`, `mov`, `avi`, `txt`). |
| `--folders` | | `all` | Specific subfolders to crawl (e.g. `uniform-light varying-light`). |
| `--dry-run` | | `False` | Scan remote directory and list matching files without downloading. |
| `--force` | | `False` | Overwrite files even if they already exist locally. |
| `--include-backup`| | `False` | Include `before_restore/` (duplicate backup). |

---

## 3. Usage Examples

Open PowerShell or Command Prompt in the repository root directory:

### Example 1: Preview Files First (Dry Run)
Before downloading gigabytes of data, you can verify which files will be fetched:
```powershell
python download_bu_dataset.py --dry-run
```
To preview only `.dat` and `.mov` files:
```powershell
python download_bu_dataset.py --ext dat mov --dry-run
```

---

### Example 2: Download Only Ground Truth (`.dat` files)
If you already have videos or only want the ground truth pose trajectories, this downloads all annotations in ~5 seconds (< 1 MB total):
```powershell
python download_bu_dataset.py --dest "data/bu_headtracking" --ext dat
```

---

### Example 3: Recommended Lightweight Setup (`.dat` + `.mov` videos)
This is the recommended setup for development. It downloads the ground truth along with compressed QuickTime video files (~200 MB total instead of ~5.5 GB uncompressed AVIs):
```powershell
python download_bu_dataset.py --dest "data/bu_headtracking" --ext dat mov
```

---

### Example 4: Full Dataset Download to a Custom Folder
Download every file (including large uncompressed `.avi` videos, ~5.5 GB total) to a specific destination folder:

```powershell
# Relative path inside the repository:
python download_bu_dataset.py --dest "data/bu_headtracking"

# Or an absolute path on any drive:
python download_bu_dataset.py --dest "D:/Datasets/bu_headtracking"
```

---

### Example 5: Download Specific Folders Only
If you only need the uniform and varying lighting test sets:
```powershell
python download_bu_dataset.py --dest "data/bu_headtracking" --folders uniform-light varying-light --ext dat mov
```

---

### Example 6: Resuming and Overwriting
- By default, the script **checks file existence and size**. If a file is already downloaded, it will be skipped automatically, allowing you to resume interrupted downloads safely.
- If you want to force re-downloading all files from scratch:
  ```powershell
  python download_bu_dataset.py --dest "data/bu_headtracking" --force
  ```

---

## 4. Understanding Ground Truth `.dat` Files

Each `.dat` file (e.g. `uniform-light/jam1.dat`) corresponds to a sequence and contains space-separated values:

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

## 5. Alternative Download Methods

### Using GNU `wget` (Git Bash / Linux / WSL)

If you prefer using command-line `wget`:

```bash
# Download .dat and .mov files into a specific directory, ignoring duplicate backup:
wget -r -np -nH --cut-dirs=1 -A "*.dat,*.mov" -X "/headtracking/before_restore" "http://csr.bu.edu/headtracking/" -P "data/bu_headtracking"
```
