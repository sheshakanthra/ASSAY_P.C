"""Train the tiny CNN baseline used as the "clean" model for every fixture.

Dataset: MedMNIST PneumoniaMNIST (28x28 grayscale chest X-rays, binary label)
if the `medmnist` package + a download are available; falls back to MNIST
(10-class) otherwise. Trains a few epochs on CPU, then saves:
  - fixtures/models/clean_cnn.pt            (torch state_dict, zip-serialized)
  - fixtures/models/clean_cnn.safetensors   (same weights, safetensors)
  - fixtures/models/eval.npz                (held-out images + labels)
  - fixtures/models/model_info.json         (architecture + dataset provenance)

CPU-only, deterministic (fixed seed). No part of this script touches the
detection path — it exists purely to produce ground-truth fixtures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file as save_safetensors
from torch import nn
from torch.utils.data import DataLoader

SEED = 0
EPOCHS = 3
BATCH_SIZE = 64
EVAL_SAMPLES = 200

FIXTURES_DIR = Path(__file__).parent
MODELS_DIR = FIXTURES_DIR / "models"
CACHE_DIR = FIXTURES_DIR / ".cache"


class TinyCNN(nn.Module):
    """Two conv blocks + a small classifier head, sized for 28x28 grayscale input."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 7 * 7, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def _load_pneumoniamnist():
    from medmnist import PneumoniaMNIST
    from torchvision import transforms

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    transform = transforms.ToTensor()
    train = PneumoniaMNIST(split="train", download=True, root=str(CACHE_DIR), transform=transform)
    test = PneumoniaMNIST(split="test", download=True, root=str(CACHE_DIR), transform=transform)
    return train, test, 2, "pneumoniamnist"


def _load_mnist():
    from torchvision import transforms
    from torchvision.datasets import MNIST

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    transform = transforms.ToTensor()
    train = MNIST(root=str(CACHE_DIR), train=True, download=True, transform=transform)
    test = MNIST(root=str(CACHE_DIR), train=False, download=True, transform=transform)
    return train, test, 10, "mnist"


def load_dataset():
    """MedMNIST PneumoniaMNIST if the package + download succeed, else MNIST."""
    try:
        return _load_pneumoniamnist()
    except Exception as e:  # noqa: BLE001 - any failure (missing pkg, no network) falls back
        print(f"PneumoniaMNIST unavailable ({e!r}); falling back to MNIST", file=sys.stderr)
        return _load_mnist()


def _labels_to_long(labels: torch.Tensor) -> torch.Tensor:
    # medmnist yields shape (batch, 1) labels; MNIST yields shape (batch,). Both squeeze cleanly.
    return labels.reshape(labels.shape[0]).long()


def train(model: nn.Module, train_ds, epochs: int) -> None:
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for images, labels in loader:
            targets = _labels_to_long(labels)
            optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"epoch {epoch + 1}/{epochs}: mean loss {total_loss / max(n_batches, 1):.4f}")


def evaluate(model: nn.Module, test_ds) -> tuple[np.ndarray, np.ndarray, float]:
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    images_out: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    with torch.no_grad():
        for images, labels in loader:
            targets = _labels_to_long(labels)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.numel()
            if sum(a.shape[0] for a in images_out) < EVAL_SAMPLES:
                images_out.append(images.numpy())
                labels_out.append(targets.numpy())
    accuracy = correct / total if total else 0.0
    eval_images = np.concatenate(images_out, axis=0)[:EVAL_SAMPLES]
    eval_labels = np.concatenate(labels_out, axis=0)[:EVAL_SAMPLES]
    return eval_images, eval_labels, accuracy


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, test_ds, num_classes, dataset_name = load_dataset()
    print(f"dataset: {dataset_name} (num_classes={num_classes}, train={len(train_ds)}, test={len(test_ds)})")

    model = TinyCNN(num_classes=num_classes)
    train(model, train_ds, EPOCHS)
    eval_images, eval_labels, accuracy = evaluate(model, test_ds)
    print(f"held-out accuracy: {accuracy:.4f}")

    state_dict = model.state_dict()
    torch.save(state_dict, str(MODELS_DIR / "clean_cnn.pt"))
    save_safetensors({k: v.contiguous() for k, v in state_dict.items()}, str(MODELS_DIR / "clean_cnn.safetensors"))

    np.savez(
        MODELS_DIR / "eval.npz",
        images=eval_images.astype(np.float32),
        labels=eval_labels.astype(np.int64),
    )

    model_info = {
        "architecture": "TinyCNN",
        "dataset": dataset_name,
        "num_classes": num_classes,
        "input_shape": [1, 28, 28],
        "held_out_accuracy": accuracy,
        "epochs": EPOCHS,
        "seed": SEED,
    }
    (MODELS_DIR / "model_info.json").write_text(json.dumps(model_info, indent=2))

    print(f"wrote {MODELS_DIR / 'clean_cnn.pt'}")
    print(f"wrote {MODELS_DIR / 'clean_cnn.safetensors'}")
    print(f"wrote {MODELS_DIR / 'eval.npz'} ({eval_images.shape[0]} samples)")
    print(f"wrote {MODELS_DIR / 'model_info.json'}")


if __name__ == "__main__":
    main()
