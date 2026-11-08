import argparse
import csv
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import json

import torch
from torchvision.utils import save_image

from weakflow.checkpoint import load_checkpoint
from weakflow.data import load_cifar10
from weakflow.model import MeanFlowNet
from weakflow.evaluation import generate, compute_fid


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--device", choices=["cuda", "cpu"], default="cuda",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        choices=["raw", "ema_099", "ema"],
        default=["raw", "ema_099", "ema"],
    )
    parser.add_argument("--nfe", nargs="+", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--images-only", action="store_true")

    args = parser.parse_args()

    saved = load_checkpoint(args.checkpoint)
    cfg = saved["cfg"]

    if saved["method"] != "weak":
        parser.error("弱形式のチェックポイントを指定してください")

    if cfg.get("smoke", False) and not args.images_only:
        parser.error(
            "smoke学習モデルは--images-onlyで確認してください"
        )

    defaults = cfg.get("evaluation", {})

    nfe_list = args.nfe or defaults.get("nfe_list", [1, 4, 16])
    n = (
        args.samples if args.samples is not None
        else defaults.get("fid_samples", 10000)
    )
    bs = (
        args.batch_size if args.batch_size is not None
        else defaults.get("batch_size", 64)
    )
    seed = (
        args.seed if args.seed is not None
        else defaults.get("seed", 42)
    )

    if min(nfe_list) < 1 or bs < 1 or n < 2:
        parser.error(
            "NFE・batch-sizeは1以上、samplesは2以上にしてください"
        )

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDAを利用できません")

    torch.set_float32_matmul_precision("high")

    x_all = None

    if not args.images_only:
        root = args.data_root or cfg.get("data_root")

        if not root:
            parser.error("--data-rootを指定してください")

        x_all = load_cifar10(root, download=False)

    ckpt = Path(args.checkpoint).resolve()
    base = (
        Path(args.output_dir)
        if args.output_dir else ckpt.parent / "evaluation"
    )

    out = base / (
        f"{ckpt.stem}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
    )
    out.mkdir(parents=True, exist_ok=False)

    metadata = dict(
        checkpoint=str(ckpt),
        step=saved["step"],
        weights=args.weights,
        nfe=nfe_list,
        samples=n,
        batch_size=bs,
        seed=seed,
        images_only=args.images_only,
    )
    (out / "settings.json").write_text(
        json.dumps(metadata, indent=2)
    )

    model = MeanFlowNet(**saved["net_cfg"]).to(device).eval()

    generator = torch.Generator(device=device).manual_seed(seed)
    e_fixed = torch.randn(
        16, 3, 32, 32,
        device=device,
        generator=generator,
    )

    state_keys = {
        "raw": "model",
        "ema_099": "ema_099",
        "ema": "ema",
    }

    with (out / "fid.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["weights", "nfe", "samples", "seed", "fid"],
        )
        writer.writeheader()

        for kind in args.weights:
            model.load_state_dict(saved[state_keys[kind]])

            for nfe in nfe_list:
                images = (
                    generate(model, e_fixed, nfe) * 0.5 + 0.5
                ).clamp(0, 1)

                save_image(
                    images,
                    out / f"{kind}_{nfe}nfe.png",
                    nrow=4,
                )

                if not args.images_only:
                    score = compute_fid(
                        model, x_all,
                        N=nfe, n=n, bs=bs, seed=seed,
                    )

                    writer.writerow(
                        dict(
                            weights=kind,
                            nfe=nfe,
                            samples=n,
                            seed=seed,
                            fid=score,
                        )
                    )
                    f.flush()

                    print(kind, nfe, score, flush=True)

    print("評価結果:", out)


if __name__ == "__main__":
    main()