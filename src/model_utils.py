import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def create_model(num_classes: int, pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return model


def save_checkpoint(
    path: Path,
    model: nn.Module,
    class_to_idx: Dict[str, int],
    image_size: int,
    arch: str = "resnet18",
) -> None:
    payload = {
        "arch": arch,
        "state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "image_size": image_size,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def write_labels_json(path: Path, class_to_idx: Dict[str, int]) -> None:
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
    ordered = [idx_to_class[idx] for idx in sorted(idx_to_class)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ordered, indent=2))


def load_checkpoint(path: Path, device: torch.device) -> Tuple[nn.Module, Dict[int, str], int]:
    checkpoint = torch.load(path, map_location=device)
    arch = checkpoint.get("arch", "resnet18")
    if arch != "resnet18":
        raise ValueError(f"Unsupported architecture in checkpoint: {arch}")
    class_to_idx = checkpoint["class_to_idx"]
    image_size = int(checkpoint.get("image_size", 224))
    model = create_model(num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
    return model, idx_to_class, image_size
