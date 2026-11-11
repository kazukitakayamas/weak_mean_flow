"""Colab entry point for the paired long-interval continuation experiment."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from weakflow.checkpoint import atomic_save, load_checkpoint
from weakflow.data import load_cifar10
from weakflow.evaluation import compute_fid, generate, prepare_real_fid
from weakflow.model import MeanFlowNet
from weakflow.trainer import train


DRIVE = Path("/content/drive/MyDrive/WeakMeanFlow")
BASE_RUN = DRIVE / "diagnosis_runs/weak_baseline_seed42_v1"
START, END = 20_000, 22_000
ARMS = {"A": 0.0, "B": 0.25}


def baseline_path(explicit):
    if explicit is not None:
        return explicit.resolve(strict=True)
    paths = sorted(BASE_RUN.glob("weak_*/step_020000.pt"))
    if not paths:
        paths = sorted(BASE_RUN.glob("weak_*/latest.pt"))
    if len(paths) != 1:
        raise ValueError(f"Pass --source PATH to select your 20k checkpoint. Candidates: {paths}")
    return paths[0].resolve()


def arm_path(source, arm):
    return source.parent / "long_interval_ab" / arm / "latest.pt"


def write_csv(path, rows):
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def prepare_arm(source, digest, arm):
    path = arm_path(source, arm)
    fork_info = dict(parent_sha256=digest, parent_step=START, arm=arm)
    if path.exists():
        state = load_checkpoint(path)
        if state["cfg"].get("interval_fork") != fork_info or state["cfg"].get("off_uniform_prob") != ARMS[arm]:
            raise ValueError(f"Existing checkpoint belongs to another experiment: {path}")
        return path, int(state["step"])

    state = load_checkpoint(source)
    if state["method"] != "weak" or state["step"] != START or state["cfg"].get("smoke"):
        raise ValueError("Select a real weak checkpoint at exactly 20,000 updates.")
    if state["cfg"].get("off_uniform_prob", 0) != 0:
        raise ValueError("The parent must use the original logit-normal off-diagonal distribution.")
    path.parent.mkdir(parents=True, exist_ok=True)
    state["cfg"].update(interval_fork=fork_info, off_uniform_prob=ARMS[arm],
                        snapshot_steps=[], download=False)
    # Preserve optimizer, both EMAs, update counts and the saved RNG states.
    atomic_save(state, path)
    return path, START


def run_training(source, args):
    with source.open("rb") as f:
        digest = hashlib.sha256()
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            digest.update(chunk)
    for arm in args.arm:
        path, step = prepare_arm(source, digest.hexdigest(), arm)
        if not START <= step <= END:
            raise ValueError(f"Unexpected step {step} at {path}")
        print(f"Arm {arm}  rho={ARMS[arm]}  step={step} -> {END}  {path}", flush=True)
        if step < END:
            train(resume=path, total_steps=END, data_root=args.data_root, device="cuda")

        state = load_checkpoint(path)
        rows = [{k: row[k] for k in ("step", "n_off", "h_ge_05", "h_ge_09")}
                for row in state["history"] if START < row["step"] <= END]
        if [r["step"] for r in rows] != list(range(START + 1, END + 1)):
            raise RuntimeError("The interval counts do not cover every continuation step.")
        write_csv(path.parent / "interval_counts.csv", rows)
        for key in ("h_ge_05", "h_ge_09"):
            n = sum(row[key] for row in rows)
            print(f"{arm} {key}: {n / len(rows):.4f} pairs/update "
                  f"({n / sum(row['n_off'] for row in rows):.6%})", flush=True)
        del state


@torch.no_grad()
def evaluate_arms(source, args):
    from torchvision.utils import save_image

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    root = source.parent / "long_interval_ab"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
    out = root / f"eval_{args.weights}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    batch = args.eval_batch
    fid = data = None
    if not args.images_only:
        data = load_cifar10(args.data_root, download=False)
        fid = prepare_real_fid(data, device, batch)

    generator = torch.Generator().manual_seed(12345)
    noise = torch.randn(128, 3, 32, 32, generator=generator)
    fid_rows, diag_rows = [], []
    parent = None
    for arm in args.arm:
        path = arm_path(source, arm)
        saved = load_checkpoint(path)
        cfg = saved["cfg"]
        lineage = cfg["interval_fork"]
        if (saved["step"] != END or lineage["arm"] != arm or lineage["parent_step"] != START
                or cfg["off_uniform_prob"] != ARMS[arm]):
            raise ValueError(f"Complete arm {arm} through step {END} before evaluating: {path}")
        if parent is not None and parent != lineage["parent_sha256"]:
            raise ValueError("A and B have different parent checkpoints")
        parent = lineage["parent_sha256"]
        model = MeanFlowNet(**saved["net_cfg"])
        model.load_state_dict(saved[{"raw": "model", "ema": "ema", "ema_099": "ema_099"}[args.weights]])
        del saved
        model = model.to(device).eval()
        squared_error = 0.
        panels = {1: [], 16: []}
        for i, e in enumerate(noise.split(batch)):
            e = e.to(device)
            x1, x16 = generate(model, e, 1), generate(model, e, 16)
            squared_error += (x1.double() - x16.double()).square().sum().item()
            # Collect the same 16 images even when eval_batch is smaller than 16.
            take = max(0, 16-i*batch)
            if take:
                for nfe, x in [(1, x1), (16, x16)]:
                    panels[nfe].append(x[:take].cpu())
        for nfe, parts in panels.items():
            save_image((torch.cat(parts) * 0.5 + 0.5).clamp(0, 1),
                       out / f"{arm}_{nfe}nfe.png", nrow=4)
        diag_rows.append(dict(arm=arm, step=END, weights=args.weights, seed=12345,
                              images=len(noise), rms_x1_vs_x16=(squared_error / noise.numel())**0.5))
        if fid is not None:
            for nfe in (1, 16):
                score = compute_fid(model, data, N=nfe, n=args.samples, bs=batch, seed=42, real_fid=fid)
                fid_rows.append(dict(arm=arm, step=END, weights=args.weights, nfe=nfe,
                                     n_real=len(data), n_fake=args.samples, seed=42, fid=score))
                write_csv(out / "fid.csv", fid_rows)
                print(f"Arm {arm}  {nfe}-NFE FID({args.samples}) = {score:.4f}", flush=True)
        del model
        write_csv(out / "diagnostics.csv", diag_rows)
    (out / "source.json").write_text(json.dumps(dict(path=str(source), sha256=parent, eval_batch=batch)))
    print("Results:", out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["train", "eval"])
    p.add_argument("--source", type=Path, help="20k parent checkpoint; default: previous Colab run")
    p.add_argument("--data-root", type=Path, default=DRIVE / "data")
    p.add_argument("--arm", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--weights", choices=["raw", "ema", "ema_099"], default="ema")
    p.add_argument("--samples", type=int, default=10_000)
    p.add_argument("--eval-batch", type=int, default=32)
    p.add_argument("--images-only", action="store_true")
    args = p.parse_args()
    if args.samples < 2 or args.eval_batch < 1:
        p.error("--samples >= 2 and --eval-batch >= 1 required")
    if not torch.cuda.is_available():
        p.error("Select a GPU runtime in Colab.")
    source = baseline_path(args.source)
    if args.action == "train":
        run_training(source, args)
    else:
        evaluate_arms(source, args)


if __name__ == "__main__":
    main()
