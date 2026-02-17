#!/usr/bin/env python3
"""
Install Stockfish binary for the current platform.

Downloads the latest Stockfish release from GitHub and extracts it to
alphazero-cpp/bin/stockfish[.exe]. Uses the official-stockfish/Stockfish
GitHub releases page.

Usage:
    python install_stockfish.py              # Download to alphazero-cpp/bin/
    python install_stockfish.py --force      # Re-download even if exists
"""

import argparse
import io
import os
import platform
import stat
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib import request, error


# Where to install the binary (relative to this script's location)
SCRIPT_DIR = Path(__file__).resolve().parent
BIN_DIR = SCRIPT_DIR.parent / "bin"

# GitHub API endpoint for latest release
RELEASES_API = "https://api.github.com/repos/official-stockfish/Stockfish/releases/latest"


def detect_platform() -> str:
    """Detect OS + architecture and return the Stockfish asset keyword.

    Stockfish release assets follow a naming pattern like:
        stockfish-windows-x86-64-avx2.zip
        stockfish-ubuntu-x86-64-avx2.tar
    We match against the broadest compatible variant (avx2 for modern CPUs).
    """
    system = platform.system().lower()
    machine = platform.machine().lower()

    if machine not in ("x86_64", "amd64"):
        print(f"ERROR: Unsupported architecture: {machine}")
        print("Only x86-64 is currently supported.")
        sys.exit(1)

    if system == "windows":
        return "windows-x86-64"
    elif system == "linux":
        return "ubuntu-x86-64"
    else:
        print(f"ERROR: Unsupported OS: {system}")
        print("Only Windows and Linux are currently supported.")
        sys.exit(1)


def get_binary_name() -> str:
    """Return the expected binary name for this platform."""
    if platform.system().lower() == "windows":
        return "stockfish.exe"
    return "stockfish"


def fetch_release_info() -> dict:
    """Fetch latest release metadata from GitHub API."""
    import json

    req = request.Request(
        RELEASES_API,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "AlphaZero-Stockfish-Installer"},
    )
    try:
        with request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except error.HTTPError as e:
        print(f"ERROR: GitHub API returned {e.code}: {e.reason}")
        sys.exit(1)
    except error.URLError as e:
        print(f"ERROR: Could not reach GitHub: {e.reason}")
        sys.exit(1)


def find_asset(release: dict, platform_key: str) -> dict:
    """Find the matching asset from release metadata.

    Prefers the avx2 build for best performance on modern CPUs.
    Falls back to the sse41-popcnt build if avx2 is not available.
    """
    assets = release.get("assets", [])

    # Try avx2 first, then sse41-popcnt, then any match
    for suffix in ("avx2", "sse41-popcnt", ""):
        for asset in assets:
            name = asset["name"].lower()
            if platform_key in name and (not suffix or suffix in name):
                return asset

    print(f"ERROR: No matching asset for platform '{platform_key}'")
    print(f"Available assets: {[a['name'] for a in assets]}")
    sys.exit(1)


def download_and_extract(asset: dict, dest_dir: Path) -> Path:
    """Download the asset archive and extract the Stockfish binary."""
    url = asset["browser_download_url"]
    name = asset["name"]
    binary_name = get_binary_name()

    print(f"Downloading: {name}")
    print(f"  URL: {url}")

    req = request.Request(url, headers={"User-Agent": "AlphaZero-Stockfish-Installer"})
    with request.urlopen(req, timeout=120) as resp:
        data = resp.read()

    print(f"  Downloaded {len(data) / 1024 / 1024:.1f} MB")

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / binary_name

    # Extract binary from archive
    if name.endswith(".zip"):
        _extract_from_zip(data, dest_path, binary_name)
    elif name.endswith(".tar") or ".tar." in name:
        _extract_from_tar(data, dest_path, binary_name)
    else:
        print(f"ERROR: Unknown archive format: {name}")
        sys.exit(1)

    # Set executable permission on Linux
    if platform.system().lower() != "windows":
        dest_path.chmod(dest_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    return dest_path


def _extract_from_zip(data: bytes, dest_path: Path, binary_name: str):
    """Extract stockfish binary from a zip archive."""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        # Find the stockfish binary inside the archive
        candidates = [n for n in zf.namelist() if n.endswith(binary_name) or n.endswith("stockfish")]
        if not candidates:
            # Some archives have the binary inside a subdirectory
            candidates = [n for n in zf.namelist() if "stockfish" in n.lower() and not n.endswith("/")]

        if not candidates:
            print(f"ERROR: Could not find stockfish binary in archive")
            print(f"Archive contents: {zf.namelist()[:20]}")
            sys.exit(1)

        # Prefer exact match, then shortest path
        candidates.sort(key=len)
        chosen = candidates[0]
        print(f"  Extracting: {chosen}")

        with zf.open(chosen) as src, open(dest_path, "wb") as dst:
            dst.write(src.read())


def _extract_from_tar(data: bytes, dest_path: Path, binary_name: str):
    """Extract stockfish binary from a tar archive."""
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        candidates = [m for m in tf.getmembers() if m.isfile() and
                       ("stockfish" in m.name.lower()) and
                       not m.name.endswith(".txt") and
                       not m.name.endswith(".md")]

        if not candidates:
            print(f"ERROR: Could not find stockfish binary in archive")
            print(f"Archive contents: {[m.name for m in tf.getmembers()[:20]]}")
            sys.exit(1)

        candidates.sort(key=lambda m: len(m.name))
        chosen = candidates[0]
        print(f"  Extracting: {chosen.name}")

        f = tf.extractfile(chosen)
        if f is None:
            print(f"ERROR: Could not extract {chosen.name}")
            sys.exit(1)

        with open(dest_path, "wb") as dst:
            dst.write(f.read())


def main():
    parser = argparse.ArgumentParser(description="Install Stockfish binary")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if binary already exists")
    args = parser.parse_args()

    binary_name = get_binary_name()
    dest_path = BIN_DIR / binary_name

    # Check if already installed
    if dest_path.exists() and not args.force:
        print(f"Stockfish already installed at: {dest_path}")
        print("Use --force to re-download.")
        return

    print("=" * 50)
    print("Stockfish Installer")
    print("=" * 50)

    platform_key = detect_platform()
    print(f"Platform: {platform_key}")

    print("Fetching latest release info...")
    release = fetch_release_info()
    tag = release.get("tag_name", "unknown")
    print(f"Latest release: {tag}")

    asset = find_asset(release, platform_key)
    dest = download_and_extract(asset, BIN_DIR)

    print()
    print(f"Stockfish installed successfully!")
    print(f"  Path: {dest}")
    print(f"  Version: {tag}")


if __name__ == "__main__":
    main()
