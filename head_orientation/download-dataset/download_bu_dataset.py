"""
Downloader & Sync Tool for the Boston University (BU) Head Tracking Dataset
Source: http://csr.bu.edu/headtracking/

Features:
- Resumable downloads via HTTP Range headers (resumes interrupted files without starting over).
- Dataset comparison mode (--compare) comparing remote vs local files, sizes, and integrity.
- Fast multi-threaded remote size checking.
- Auto-detection of existing dataset directories (e.g. inside `download-dataset`).
- Graceful Ctrl+C interruption with clean file closing.
- Pure Python standard library (no pip dependencies required).
"""

import os
import re
import sys
import time
import argparse
import urllib.request
import urllib.parse
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor


BASE_URL = "http://csr.bu.edu/headtracking/"


def detect_default_dest():
    """Smart detection of default destination directory."""
    candidates = [
        os.path.join("head_orientation", "download-dataset", "dataset"),
        os.path.join("head_orientation", "download-dataset"),
        os.path.join("download-dataset", "dataset"),
        "download-dataset",
        "dataset",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "uniform-light")):
            return c

    # Fallback based on script location or current directory
    if os.path.basename(os.getcwd()) == "download-dataset":
        if os.path.exists("dataset"):
            return "dataset"
        return "."
    if os.path.exists(os.path.join("head_orientation", "download-dataset")):
        return os.path.join("head_orientation", "download-dataset")
    return "download-dataset"


class DirectoryIndexParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            for name, value in attrs:
                if name.lower() == "href" and value:
                    self.links.append(value)


def get_directory_links(url):
    """Fetch HTML directory index and return relative links."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8", errors="ignore")

    parser = DirectoryIndexParser()
    parser.feed(html)

    valid_links = []
    for link in parser.links:
        if link.startswith("?") or link.startswith("#") or link == "/" or "Parent Directory" in link:
            continue
        if link.startswith("/"):
            continue
        valid_links.append(link)
    return valid_links


def format_size(bytes_num):
    if bytes_num is None:
        return "Unknown"
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_num < 1024.0:
            return f"{bytes_num:.1f} {unit}"
        bytes_num /= 1024.0
    return f"{bytes_num:.1f} TB"


def get_remote_file_size(file_url, timeout=10):
    """Retrieve Content-Length via HTTP HEAD request."""
    req = urllib.request.Request(
        file_url,
        method="HEAD",
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            cl = resp.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception:
        return None


def crawl_and_collect(base_url, rel_path="", allowed_exts=None, skip_dirs=None):
    """Recursively collect list of (remote_url, relative_local_path)."""
    current_url = urllib.parse.urljoin(base_url, rel_path)
    if not current_url.endswith("/"):
        current_url += "/"

    items = []
    try:
        links = get_directory_links(current_url)
    except Exception as e:
        print(f"Could not list directory {current_url}: {e}")
        return items

    for link in links:
        if link.endswith("/"):
            dir_name = link.rstrip("/")
            if skip_dirs and dir_name in skip_dirs:
                continue
            sub_rel = f"{rel_path}{link}" if rel_path else link
            items.extend(crawl_and_collect(base_url, sub_rel, allowed_exts, skip_dirs))
        else:
            _, ext = os.path.splitext(link)
            ext = ext.lower().lstrip(".")
            if allowed_exts and ext not in allowed_exts and "all" not in allowed_exts:
                continue
            file_url = urllib.parse.urljoin(current_url, link)
            local_rel = os.path.join(rel_path, link) if rel_path else link
            items.append((file_url, local_rel))

    return items


def compare_remote_vs_local(files_to_download, dest_dir, max_workers=8):
    """
    Compare remote files against local filesystem.
    Returns dict categorized into: 'complete', 'partial', 'missing', 'corrupt'.
    """
    print(f"\nComparing {len(files_to_download)} files with local destination [{dest_dir}]...")
    dest_dir = os.path.abspath(dest_dir)

    def check_file(item):
        remote_url, rel_path = item
        local_path = os.path.join(dest_dir, rel_path)

        if not os.path.exists(local_path):
            # Fast check: not on disk
            return {
                "status": "missing",
                "rel_path": rel_path,
                "remote_url": remote_url,
                "local_size": 0,
                "remote_size": None
            }

        local_size = os.path.getsize(local_path)
        # Fetch remote size to verify integrity
        remote_size = get_remote_file_size(remote_url)

        if remote_size is None:
            status = "complete" if local_size > 0 else "missing"
        elif local_size == remote_size:
            status = "complete"
        elif 0 < local_size < remote_size:
            status = "partial"
        elif local_size > remote_size:
            status = "corrupt"
        else:
            status = "missing"

        return {
            "status": status,
            "rel_path": rel_path,
            "remote_url": remote_url,
            "local_size": local_size,
            "remote_size": remote_size
        }

    results = {"complete": [], "partial": [], "missing": [], "corrupt": []}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, res in enumerate(executor.map(check_file, files_to_download), 1):
            results[res["status"]].append(res)
            sys.stdout.write(f"\r  Audited {idx}/{len(files_to_download)} files...")
            sys.stdout.flush()

    sys.stdout.write("\n")
    return results


def print_comparison_report(results, dest_dir):
    """Print structured summary of local vs remote status."""
    complete_bytes = sum(f["local_size"] for f in results["complete"])
    partial_bytes = sum(f["local_size"] for f in results["partial"])

    print("\n" + "=" * 65)
    print("DATASET COMPARISON REPORT (Remote vs Local)")
    print(f"Target Directory : {os.path.abspath(dest_dir)}")
    print("=" * 65)
    print(f"  [OK]      Complete files      : {len(results['complete']):4d}  ({format_size(complete_bytes)})")
    print(f"  [PARTIAL] Partially downloaded: {len(results['partial']):4d}  ({format_size(partial_bytes)} saved, needs resume)")
    print(f"  [ALERT]   Corrupt / Oversized : {len(results['corrupt']):4d}")
    print(f"  [MISSING] Missing files       : {len(results['missing']):4d}")
    print("-" * 65)
    total_files = sum(len(v) for v in results.values())
    pct = (len(results["complete"]) / total_files * 100) if total_files > 0 else 0
    print(f"  Overall Progress         : {len(results['complete'])}/{total_files} files ({pct:.1f}%)")
    print("=" * 65)

    if results["partial"]:
        print("\nPartially downloaded files (will resume via HTTP Range):")
        for f in results["partial"][:10]:
            print(f"  - {f['rel_path']}: {format_size(f['local_size'])} / {format_size(f['remote_size'])}")
        if len(results["partial"]) > 10:
            print(f"    ... and {len(results['partial']) - 10} more.")

    if results["missing"]:
        print(f"\nMissing files to download ({len(results['missing'])} total):")
        for f in results["missing"][:10]:
            print(f"  - {f['rel_path']}")
        if len(results["missing"]) > 10:
            print(f"    ... and {len(results['missing']) - 10} more.")
    print()


def download_file(file_url, dest_path, force=False):
    """Download a file with HTTP Range resume support."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    local_size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0

    # 1. Check remote file size
    remote_size = get_remote_file_size(file_url)

    # 2. Skip if already complete
    if not force and local_size > 0:
        if remote_size is not None and local_size == remote_size:
            print(f"  [Skipped] {os.path.basename(dest_path)} (already complete: {format_size(local_size)})")
            return True
        elif remote_size is None and local_size > 0:
            print(f"  [Skipped] {os.path.basename(dest_path)} (already exists: {format_size(local_size)})")
            return True

    # 3. Setup resume headers if partial
    resume = False
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    if not force and 0 < local_size < (remote_size or float('inf')):
        headers["Range"] = f"bytes={local_size}-"
        resume = True

    req = urllib.request.Request(file_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            status = getattr(response, "status", 200)

            if resume and status == 206:
                mode = "ab"
                downloaded = local_size
                total_expected = remote_size
                print(f"  [Resuming] {os.path.basename(dest_path)} from {format_size(local_size)} / {format_size(total_expected)} (remaining: {format_size(total_expected - local_size)})")
            else:
                mode = "wb"
                downloaded = 0
                cl = response.headers.get("Content-Length")
                total_expected = int(cl) if cl else remote_size
                action = "Re-downloading" if local_size > 0 else "Downloading"
                print(f"  [{action}] {os.path.basename(dest_path)} " +
                      (f"({format_size(total_expected)})" if total_expected else ""))

            block_size = 64 * 1024
            start_time = time.time()
            chunk_bytes = 0

            with open(dest_path, mode) as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    chunk_bytes += len(chunk)
                    downloaded += len(chunk)
                    if total_expected:
                        percent = (downloaded / total_expected) * 100
                        elapsed = time.time() - start_time
                        speed = chunk_bytes / (elapsed + 1e-5)
                        sys.stdout.write(f"\r    -> {percent:5.1f}% | {format_size(downloaded)} / {format_size(total_expected)} | {format_size(speed)}/s")
                        sys.stdout.flush()

            if total_expected:
                sys.stdout.write("\n")
            return True

    except KeyboardInterrupt:
        print(f"\n  [Paused] Download interrupted by user. File state saved at: {format_size(os.path.getsize(dest_path))}")
        raise
    except Exception as e:
        print(f"\n  [Error] Failed to download {file_url}: {e}")
        return False


def main():
    default_dest = detect_default_dest()

    parser = argparse.ArgumentParser(
        description="Download, sync, and resume the Boston University Head Tracking Dataset."
    )
    parser.add_argument(
        "--dest", "-d",
        default=default_dest,
        help=f"Destination directory (default: {default_dest})"
    )
    parser.add_argument(
        "--ext",
        nargs="+",
        default=["all"],
        help="Filter file extensions (e.g. --ext dat mov, or --ext all [default: all])"
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=None,
        help="Specific subfolders to crawl (e.g. --folders uniform-light varying-light)"
    )
    parser.add_argument(
        "--compare", "--status",
        dest="compare",
        action="store_true",
        help="Compare remote vs local dataset: audit completed, partial, and missing files without downloading."
    )
    parser.add_argument(
        "--include-backup",
        action="store_true",
        help="Include 'before_restore/' directory (duplicate backup, skipped by default)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and list files to download without actually downloading"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file already exists locally"
    )

    args = parser.parse_args()

    allowed_exts = set(e.lower().lstrip(".") for e in args.ext)
    skip_dirs = set()
    if not args.include_backup:
        skip_dirs.add("before_restore")

    print("=" * 65)
    print("Boston University Head Tracking Dataset Tool")
    print(f"Source URL  : {BASE_URL}")
    print(f"Destination : {os.path.abspath(args.dest)}")
    print(f"Extensions  : {', '.join(allowed_exts)}")
    if args.folders:
        print(f"Folders     : {', '.join(args.folders)}")
    print("=" * 65)

    print("Scanning remote directories...")
    files_to_download = []
    if args.folders:
        for folder in args.folders:
            folder_clean = folder.strip("/") + "/"
            files = crawl_and_collect(BASE_URL, folder_clean, allowed_exts, skip_dirs)
            files_to_download.extend(files)
    else:
        files_to_download = crawl_and_collect(BASE_URL, "", allowed_exts, skip_dirs)

    print(f"Found {len(files_to_download)} remote files matching filter criteria.")

    if args.dry_run:
        print("\n--- Dry Run List ---")
        for remote_url, local_rel in files_to_download:
            print(f"  {local_rel} -> {remote_url}")
        print(f"\nDry run complete. {len(files_to_download)} files listed.")
        return

    # Comparison / Status Mode
    if args.compare:
        results = compare_remote_vs_local(files_to_download, args.dest)
        print_comparison_report(results, args.dest)
        return

    dest_dir = os.path.abspath(args.dest)
    os.makedirs(dest_dir, exist_ok=True)

    success_count = 0
    fail_count = 0

    print(f"\nStarting sync/download to: {dest_dir}")
    print("Files already downloaded will be skipped; partially downloaded files will resume.")
    print("Press Ctrl+C at any time to pause safely.\n")

    try:
        for idx, (remote_url, local_rel) in enumerate(files_to_download, 1):
            target_path = os.path.join(dest_dir, local_rel)
            print(f"[{idx}/{len(files_to_download)}]")
            ok = download_file(remote_url, target_path, force=args.force)
            if ok:
                success_count += 1
            else:
                fail_count += 1
    except KeyboardInterrupt:
        print("\n" + "!" * 65)
        print("Download paused by user.")
        print("To resume from where you left off, simply re-run:")
        print(f"  python {sys.argv[0]} --dest \"{args.dest}\"")
        print("!" * 65)
        return

    print("\n" + "=" * 65)
    print(f"Sync complete: {success_count} succeeded, {fail_count} failed.")
    print(f"Saved to: {dest_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
