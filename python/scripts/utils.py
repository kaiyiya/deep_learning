import json
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_data_dir() -> Path:
    return get_project_root() / "data"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_image_files(root: Path) -> List[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in extensions)


def load_embedding_model(device: torch.device) -> torch.nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model


def build_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def extract_feature(
    model: torch.nn.Module,
    transform: transforms.Compose,
    image_path: Path,
    device: torch.device,
) -> np.ndarray:
    with Image.open(image_path).convert("RGB") as img:
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(tensor)
        feature = embedding.cpu().numpy().reshape(-1)
    norm = np.linalg.norm(feature)
    if norm > 0:
        feature = feature / norm
    return feature.astype(np.float32)


def save_index(embeddings: np.ndarray, metadata: List[dict], output_dir: Path) -> None:
    ensure_dir(output_dir)
    np.savez_compressed(output_dir / "index.npz", embeddings=embeddings)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_index(artifacts_dir: Path) -> Tuple[np.ndarray, List[dict]]:
    index_path = artifacts_dir / "index.npz"
    metadata_path = artifacts_dir / "metadata.json"
    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"未找到索引文件：{index_path} 或 {metadata_path}")
    embeddings = np.load(index_path)["embeddings"]
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return embeddings.astype(np.float32), metadata


def cosine_similarity(query: np.ndarray, items: np.ndarray) -> np.ndarray:
    if query.ndim != 1:
        raise ValueError("query 向量必须是一维")
    if items.ndim != 2:
        raise ValueError("items 必须是二维矩阵")
    query_norm = np.linalg.norm(query)
    if not math.isclose(query_norm, 1.0, rel_tol=1e-3):
        query = query / max(query_norm, 1e-12)
    item_norms = np.linalg.norm(items, axis=1, keepdims=True)
    normalized = items / np.maximum(item_norms, 1e-12)
    return (normalized @ query).reshape(-1)

