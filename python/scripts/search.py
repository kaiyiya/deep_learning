import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .utils import (
    build_transform,
    cosine_similarity,
    extract_feature,
    get_data_dir,
    load_embedding_model,
    load_index,
)


def prepare_query_embedding(image_path: Path, device: torch.device) -> np.ndarray:
    if not image_path.exists():
        raise FileNotFoundError(f"查询图像 {image_path} 不存在")
    model = load_embedding_model(device)
    transform = build_transform()
    feature = extract_feature(model, transform, image_path, device)
    return feature


def main() -> None:
    parser = argparse.ArgumentParser(description="基于向量索引执行图像检索")
    parser.add_argument("--query", type=str, required=True, help="查询图像路径")
    parser.add_argument(
        "--topk",
        type=int,
        default=12,
        help="返回的相似图像数量",
    )
    parser.add_argument(
        "--artifacts",
        type=str,
        default=None,
        help="索引文件所在目录，默认使用 data/artifacts",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="指定推理设备，默认自动检测",
    )
    args = parser.parse_args()

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    artifacts_dir = Path(args.artifacts).resolve() if args.artifacts else get_data_dir() / "artifacts"
    embeddings, metadata = load_index(artifacts_dir)

    query_path = Path(args.query).resolve()
    query_feature = prepare_query_embedding(query_path, device)

    similarities = cosine_similarity(query_feature, embeddings)
    topk = min(args.topk, similarities.shape[0])
    indices = np.argpartition(-similarities, range(topk))[:topk]
    sorted_idx = indices[np.argsort(-similarities[indices])]

    results = []
    for rank, idx in enumerate(sorted_idx, start=1):
        item = metadata[idx]
        results.append(
            {
                "rank": rank,
                "score": float(similarities[idx]),
                **item,
            }
        )

    payload = {
        "query": str(query_path),
        "results": results,
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()

