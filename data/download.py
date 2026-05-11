from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "Ink0722/Real-UI-Clickboxes"
REPO_TYPE = "dataset"
DEFAULT_TARGET_DIR = Path(__file__).resolve().parent / "Real-UI-Clickboxes"


def download_dataset(target_dir: Path, revision: str | None = None) -> Path:
    """Download the dataset snapshot into the requested local directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        revision=revision,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )
    return target_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the Real-UI-Clickboxes dataset into the project's data directory.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help=f"Local directory to store the dataset (default: {DEFAULT_TARGET_DIR}).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional branch, tag, or commit hash to download.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    target_dir = args.target_dir.resolve()
    print(f"Downloading {REPO_ID} to {target_dir} ...")
    download_dataset(target_dir=target_dir, revision=args.revision)
    print("Download completed.")
    print(f"Dataset is available at: {target_dir}")


if __name__ == "__main__":
    main()

