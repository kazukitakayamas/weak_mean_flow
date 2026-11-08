from pathlib import Path
import pickle

import numpy as np
import torch


def load_cifar10(root, download=False):
    root = Path(root)
    found = sorted(root.rglob("data_batch_1"))

    if not found and download:
        from torchvision.datasets import CIFAR10

        CIFAR10(root=str(root), train=True, download=True)
        found = sorted(root.rglob("data_batch_1"))

    if len(found) != 1:
        raise FileNotFoundError(
            f"{root}: data_batch_1 が {len(found)} 個あります。"
            "展開済みCIFAR-10が1組だけある場所を指定してください。"
        )

    folder = found[0].parent
    batches = []

    for i in range(1, 6):
        with (folder / f"data_batch_{i}").open("rb") as f:
            batches.append(
                pickle.load(f, encoding="bytes")[b"data"]
            )

    x = torch.from_numpy(
        np.concatenate(batches).reshape(-1, 3, 32, 32)
    )

    if x.dtype != torch.uint8 or tuple(x.shape) != (50000, 3, 32, 32):
        raise ValueError("CIFAR-10の形状またはdtypeが想定と異なります")

    return x


def sample_data(x_all, n, device):
    # ノートブックと同じく、CPUの乱数で復元抽出する。
    indices = torch.randint(len(x_all), (n,))

    return (
        x_all[indices].to(device).float()
        / 255.0 * 2.0 - 1.0
    )