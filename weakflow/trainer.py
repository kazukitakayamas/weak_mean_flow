import copy
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import time

import torch
import yaml

from .model import MeanFlowNet
from .data import load_cifar10
from .time_sampling import draw_comparison_batch
from .loss import backward_comparison
from .ema import EMA
from .checkpoint import (
    load_checkpoint,
    restore_checkpoint,
    save_checkpoint,
)


def train(
    cfg=None,
    resume=None,
    total_steps=None,
    data_root=None,
    device=None,
    smoke=False,
):
    saved = load_checkpoint(resume) if resume else None
    cfg = copy.deepcopy(saved["cfg"] if saved else cfg)

    if not isinstance(cfg, dict):
        raise ValueError("設定を辞書として読み込めませんでした")

    net_cfg = copy.deepcopy(
        saved["net_cfg"] if saved else cfg["model"]
    )

    # 動作確認用。研究結果を比較する実験には使用しない。
    if smoke:
        if saved:
            raise ValueError(
                "再開時は--smoke不要です。保存済み設定を使います"
            )

        cfg.update(
            smoke=True,
            batch_size=8,
            micro_batch=4,
            features=8,
            log_every=1,
            save_every=1,
            snapshot_steps=[],
        )
        net_cfg.update(model_channels=32, num_blocks=1)

    smoke = cfg.get("smoke", False)
    device = torch.device(
        device or ("cpu" if smoke else "cuda")
    )

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDAを利用できません。DockerのGPU設定を確認してください"
        )

    torch.set_float32_matmul_precision("high")

    total_steps = int(
        total_steps if total_steps is not None
        else (3 if smoke else cfg.get("total_steps", 20000))
    )
    start = saved["step"] if saved else 0

    if total_steps <= start:
        raise ValueError(
            f"total_stepsは保存済みstep={start}より大きくしてください"
        )

    if cfg["batch_size"] < 4 or cfg["batch_size"] % 2:
        raise ValueError("batch_sizeは4以上の偶数にしてください")

    for key in ("micro_batch", "features", "log_every", "save_every"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key}は正の整数にしてください")

    if (
        cfg["warmup_steps"] < 0
        or cfg["time_std"] <= 0
        or cfg["norm_eps"] <= 0
    ):
        raise ValueError(
            "warmup_steps/time_std/norm_epsの値が不正です"
        )

    if (
        net_cfg.get("dropout") != 0
        or net_cfg.get("augment_dim") != 0
    ):
        raise ValueError(
            "今回の移植版はdropout=0、augment_dim=0を使用します"
        )

    cfg["total_steps"] = total_steps

    if data_root is not None:
        cfg["data_root"] = str(data_root)

    if smoke:
        x_all = torch.randint(
            0, 256, (512, 3, 32, 32),
            dtype=torch.uint8,
            generator=torch.Generator().manual_seed(0),
        )
    else:
        if not cfg.get("data_root"):
            raise ValueError(
                "--data-rootでCIFAR-10の場所を指定してください"
            )

        x_all = load_cifar10(
            cfg["data_root"],
            cfg.get("download", False),
        )

    # ノートブックと同じ順序でモデルを初期化する。
    torch.manual_seed(cfg["seed"])

    model = MeanFlowNet(**net_cfg).to(device).train()
    ema = EMA(model, cfg["ema_decay"], period=1)
    ema_099 = EMA(model, cfg["ema_decay_099"], period=1)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        betas=tuple(cfg["betas"]),
        weight_decay=0,
    )

    feature_rng = torch.Generator(device=device)
    feature_rng.manual_seed(cfg["seed"] + 10000)

    if saved:
        restore_checkpoint(
            saved, model, ema, ema_099, opt, feature_rng, device,
        )

        history = saved["history"]
        run_dir = Path(resume).resolve().parent

    else:
        history = []
        name = cfg["experiment_name"] + ("_smoke" if smoke else "")

        run_dir = Path(cfg["output_root"]) / name / (
            f"weak_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)

    del saved

    # 再開時も、その実行で使った設定を記録する。
    with (run_dir / f"config_until_{total_steps:06d}.yaml").open("w") as f:
        yaml.safe_dump(
            {**cfg, "model": net_cfg},
            f,
            sort_keys=False,
        )

    print("保存先:", run_dir, flush=True)
    started = time.perf_counter()

    for step in range(start + 1, total_steps + 1):
        warmup = cfg["warmup_steps"]
        lr = cfg["lr"] * (
            min((step - 1) / warmup, 1.0) if warmup else 1.0
        )

        for group in opt.param_groups:
            group["lr"] = lr

        batch = draw_comparison_batch(cfg, x_all, device)

        opt.zero_grad(set_to_none=True)

        metrics = backward_comparison(
            model, "weak", batch, cfg, feature_rng,
        )

        if not bool(torch.isfinite(metrics).all()):
            raise FloatingPointError(
                f"step={step}: 損失が非有限値です"
            )

        # 全マイクロバッチの逆伝播後に1回だけ更新する。
        opt.step()
        ema.update(model)
        ema_099.update(model)

        objective, diag, off = metrics.cpu().tolist()

        history.append(
            dict(
                step=step,
                objective=objective,
                diag=diag,
                off=off,
                lr=lr,
            )
        )

        if step == 1 or step % cfg["log_every"] == 0:
            recent = history[-cfg["log_every"]:]

            avg_diag = sum(row["diag"] for row in recent) / len(recent)
            avg_off = sum(row["off"] for row in recent) / len(recent)

            print(
                f"[weak] step={step:6d} "
                f"diag={avg_diag:.4e} "
                f"off={avg_off:+.4e} "
                f"lr={lr:.2e}",
                flush=True,
            )

        if (
            step % cfg["save_every"] == 0
            or step == total_steps
            or step in cfg.get("snapshot_steps", [])
        ):
            save_checkpoint(
                run_dir, step, cfg, net_cfg, model, ema, ema_099,
                opt, history, feature_rng, device,
            )

    print(f"今回の実行時間: {time.perf_counter() - started:.1f} 秒")
    print("チェックポイント:", run_dir / "latest.pt")

    return run_dir