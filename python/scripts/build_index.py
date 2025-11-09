import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
import torch

from .download_dataset import ensure_dataset
from .utils import (
    build_transform,
    ensure_dir,
    extract_feature,
    get_data_dir,
    load_embedding_model,
    list_image_files,
    save_index,
)


def prepare_images(dataset_root: Path) -> List[Path]:
    images = list_image_files(dataset_root)
    if not images:
        raise RuntimeError(f"数据集目录 {dataset_root} 中未找到图像文件")
    return images


def compute_embeddings(
    image_paths: List[Path],
    device: torch.device,
) -> Tuple[np.ndarray, List[dict]]:
    model = load_embedding_model(device)
    transform = build_transform()

    features = []
    metadata = []

    for image_path in tqdm(image_paths, desc="提取特征"):
        feature = extract_feature(model, transform, image_path, device)
        features.append(feature)

        relative_path = image_path.relative_to(get_data_dir())
        metadata.append(
            {
                "relative_path": str(relative_path).replace("\\", "/"),
                "label": image_path.parent.name,
                "filename": image_path.name,
            }
        )

    embeddings = np.vstack(features)
    return embeddings, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="构建图像检索特征索引")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="数据集根目录，默认使用 data/dataset/flower_photos",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="索引输出目录，默认使用 data/artifacts",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="缺少数据时自动下载示例数据集",
    )
    args = parser.parse_args()

    data_root = get_data_dir()

    if args.dataset_root:
        dataset_root = Path(args.dataset_root).resolve()
    else:
        dataset_root = data_root / "dataset" / "flower_photos"

    if args.force_download or not dataset_root.exists():
        dataset_root = ensure_dataset(root_dir=data_root, force=args.force_download)

    if not dataset_root.exists():
        raise FileNotFoundError(f"数据集目录 {dataset_root} 不存在，请先下载或指定路径")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    image_paths = prepare_images(dataset_root)
    embeddings, metadata = compute_embeddings(image_paths, device)

    output_dir = Path(args.output).resolve() if args.output else data_root / "artifacts"
    ensure_dir(output_dir)
    save_index(embeddings, metadata, output_dir)

    stats = {
        "total_images": len(metadata),
        "embedding_dim": int(embeddings.shape[1]),
        "device": str(device),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
        import json

        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("索引构建完成")
    print(stats)


if __name__ == "__main__":
    main()

