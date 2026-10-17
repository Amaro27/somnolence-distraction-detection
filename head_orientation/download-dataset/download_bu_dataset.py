"""
Downloader for the Boston University (BU) Head Tracking Dataset
Source: http://csr.bu.edu/headtracking/

Supports:
- Recursive directory traversal of the Apache directory index.
- Choosing destination folder.
- Filtering by file extension (.dat, .mov, .avi, etc.).
- Excluding redundant backup folder (`before_restore/`).
- Resuming / skipping existing files.
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


BASE_URL = "http://csr.bu.edu/headtracking/"
DEFAULT_DEST = os.path.join("data", "bu_headtracking")


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
        # Ignore query params, anchors, parent directory links
        if link.startswith("?") or link.startswith("#") or link == "/" or "Parent Directory" in link:
            continue
        if link.startswith("/"):
            continue
        valid_links.append(link)
    return valid_links


def format_size(bytes_num):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_num < 1024.0:
            return f"{bytes_num:.1f} {unit}"
        bytes_num /= 1024.0
    return f"{bytes_num:.1f} TB"


def download_file(file_url, dest_path, force=False):
    """Download a file with progress display."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # Check remote file size if possible
    req = urllib.request.Request(
        file_url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            content_length = response.headers.get("Content-Length")
            total_size = int(content_length) if content_length else None

            if not force and os.path.exists(dest_path):
                local_size = os.path.getsize(dest_path)
                if total_size is not None and local_size == total_size:
                    print(f"  [Skipped] {os.path.basename(dest_path)} (already complete: {format_size(local_size)})")
                    return True
                elif total_size is None and local_size > 0:
                    print(f"  [Skipped] {os.path.basename(dest_path)} (already exists: {format_size(local_size)})")
                    return True

            print(f"  [Downloading] {os.path.basename(dest_path)} " +
                  (f"({format_size(total_size)})" if total_size else ""))

            block_size = 64 * 1024
            downloaded = 0
            start_time = time.time()

            with open(dest_path, "wb") as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        percent = (downloaded / total_size) * 100
                        speed = downloaded / (time.time() - start_time + 1e-5)
                        sys.stdout.write(f"\r    -> {percent:5.1f}% | {format_size(downloaded)} / {format_size(total_size)} | {format_size(speed)}/s")
                        sys.stdout.flush()
            if total_size:
                sys.stdout.write("\n")
            return True
    except Exception as e:
        print(f"\n  [Error] Failed to download {file_url}: {e}")
        return False


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


def main():
    parser = argparse.ArgumentParser(description="Download the Boston University Head Tracking Dataset")
    parser.add_argument(
        "--dest", "-d",
        default=DEFAULT_DEST,
        help=f"Destination directory (default: {DEFAULT_DEST})"
    )
    parser.add_argument(
        "--ext",
        nargs="+",
        default=["all"],
        help="Filter file extensions (e.g. --ext dat mov, or --ext dat, or --ext all [default: all])"
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=None,
        help="Specify specific subfolders to download (e.g. --folders uniform-light varying-light)"
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

    print("=" * 60)
    print("Boston University Head Tracking Dataset Downloader")
    print(f"Source URL : {BASE_URL}")
    print(f"Destination: {os.path.abspath(args.dest)}")
    print(f"Extensions : {', '.join(allowed_exts)}")
    if args.folders:
        print(f"Folders    : {', '.join(args.folders)}")
    print("=" * 60)

    print("Scanning remote directories (this may take a few seconds)...")
    files_to_download = []

    if args.folders:
        for folder in args.folders:
            folder_clean = folder.strip("/") + "/"
            files = crawl_and_collect(BASE_URL, folder_clean, allowed_exts, skip_dirs)
            files_to_download.extend(files)
    else:
        files_to_download = crawl_and_collect(BASE_URL, "", allowed_exts, skip_dirs)

    print(f"\nFound {len(files_to_download)} files matching criteria.")

    if args.dry_run:
        print("\n--- Dry Run List ---")
        for remote_url, local_rel in files_to_download:
            print(f"  {local_rel} -> {remote_url}")
        print(f"\nDry run complete. {len(files_to_download)} files listed.")
        return

    dest_dir = os.path.abspath(args.dest)
    os.makedirs(dest_dir, exist_ok=True)

    success_count = 0
    fail_count = 0

    print("\nStarting download...\n")
    for idx, (remote_url, local_rel) in enumerate(files_to_download, 1):
        target_path = os.path.join(dest_dir, local_rel)
        print(f"[{idx}/{len(files_to_download)}]")
        ok = download_file(remote_url, target_path, force=args.force)
        if ok:
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 60)
    print(f"Download complete: {success_count} succeeded, {fail_count} failed.")
    print(f"Saved to: {dest_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
