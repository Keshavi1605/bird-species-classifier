import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.model_utils import create_model, get_transforms, save_checkpoint, write_labels_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train bird species classifier.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/Users/kesh/Documents/Projects/Bird Species Detection/100-bird-species"),
        help="Dataset root with train/, valid/, test/ folders.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Model output directory.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size.")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument(
        "--unfreeze-all",
        action="store_true",
        help="Train full backbone. Default trains classifier head only for stability/speed.",
    )
    return parser.parse_args()


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / max(total, 1), correct / max(total, 1)


def build_dataloaders(data_dir: Path, image_size: int, batch_size: int, workers: int) -> Dict[str, DataLoader]:
    train_path = data_dir / "train"
    valid_path = data_dir / "valid"
    test_path = data_dir / "test"
    missing = [p for p in [train_path, valid_path, test_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset folders: {missing}")

    train_ds = ImageFolder(train_path, transform=get_transforms(image_size, train=True))
    valid_ds = ImageFolder(valid_path, transform=get_transforms(image_size, train=False))
    test_ds = ImageFolder(test_path, transform=get_transforms(image_size, train=False))

    if train_ds.class_to_idx != valid_ds.class_to_idx or train_ds.class_to_idx != test_ds.class_to_idx:
        raise ValueError("train/valid/test class mappings do not match.")

    print(f"Detected classes: {len(train_ds.classes)}")
    print(f"Train samples: {len(train_ds)} | Valid samples: {len(valid_ds)} | Test samples: {len(test_ds)}")

    return {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "valid": DataLoader(
            valid_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "class_to_idx": train_ds.class_to_idx,
    }


def main() -> None:
    args = parse_args()
    device = choose_device()
    print(f"Using device: {device}")

    loaders = build_dataloaders(args.data_dir, args.image_size, args.batch_size, args.workers)
    class_to_idx = loaders["class_to_idx"]

    model = create_model(num_classes=len(class_to_idx), pretrained=True)
    if not args.unfreeze_all:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    model = model.to(device)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = -1.0
    best_path = args.output_dir / "bird_classifier.pt"
    labels_path = args.output_dir / "labels.json"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, loaders["train"], optimizer, device)
        val_loss, val_acc = evaluate(model, loaders["valid"], device)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(best_path, model, class_to_idx, args.image_size)
            write_labels_json(labels_path, class_to_idx)
            print(f"Saved best model to {best_path} (val_acc={val_acc:.4f})")

    model.load_state_dict(torch.load(best_path, map_location=device)["state_dict"])
    test_loss, test_acc = evaluate(model, loaders["test"], device)
    print(f"Final test_loss={test_loss:.4f} test_acc={test_acc:.4f}")
    print(f"Labels saved to {labels_path}")


if __name__ == "__main__":
    main()
