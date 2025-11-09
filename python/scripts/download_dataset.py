import argparse
import tarfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

from .utils import get_data_dir, ensure_dir

DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
ARCHIVE_NAME = "flower_photos.tgz"
TARGET_DIRNAME = "flower_photos"


def download_file(url: str, target_path: Path, chunk_size: int = 1024 * 1024) -> None:
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(target_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=f"下载 {target_path.name}"
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def extract_archive(archive_path: Path, target_dir: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=target_dir)


def ensure_dataset(root_dir: Optional[Path] = None, force: bool = False) -> Path:
    data_dir = root_dir or get_data_dir()
    dataset_dir = data_dir / "dataset" / TARGET_DIRNAME
    if dataset_dir.exists() and not force:
        return dataset_dir

    ensure_dir(data_dir / "dataset")
    ensure_dir(data_dir / "artifacts")

    archive_path = data_dir / "artifacts" / ARCHIVE_NAME
    if force or not archive_path.exists():
        download_file(DATASET_URL, archive_path)

    extract_archive(archive_path, data_dir / "dataset")
    return dataset_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="下载示例花卉图像数据集")
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载并覆盖现有数据",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="自定义数据根目录，默认使用项目 data 目录",
    )
    args = parser.parse_args()

    root_dir = Path(args.root).resolve() if args.root else None
    dataset_path = ensure_dataset(root_dir=root_dir, force=args.force)
    print(f"数据集可用：{dataset_path}")


if __name__ == "__main__":
    main()

