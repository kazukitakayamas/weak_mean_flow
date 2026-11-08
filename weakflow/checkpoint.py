from pathlib import Path
import torch


def load_checkpoint(path):
    # 自分で作成したチェックポイントを読み込む。
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def atomic_save(state, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")

    torch.save(state, temporary)
    temporary.replace(path)


def save_checkpoint(
    run_dir, step, cfg, net_cfg, model, ema, ema_099,
    opt, history, feature_rng, device,
):
    state = dict(
        method="weak",
        step=step,
        cfg=cfg,
        net_cfg=net_cfg,
        model=model.state_dict(),
        ema=ema.net.state_dict(),
        ema_num_updates=ema.num_updates,
        ema_099=ema_099.net.state_dict(),
        ema_099_num_updates=ema_099.num_updates,
        optimizer=opt.state_dict(),
        history=history,
        feature_rng=feature_rng.get_state(),
        cpu_rng=torch.get_rng_state(),
        cuda_rng=(
            [torch.cuda.get_rng_state(device)]
            if device.type == "cuda" else None
        ),
    )

    run_dir = Path(run_dir)
    atomic_save(state, run_dir / "latest.pt")

    if step in cfg.get("snapshot_steps", [2000, 5000, 10000, 20000]):
        atomic_save(state, run_dir / f"step_{step:06d}.pt")


def restore_checkpoint(
    saved, model, ema, ema_099, opt, feature_rng, device,
):
    if saved["method"] != "weak":
        raise ValueError("弱形式のチェックポイントではありません")

    cuda_rng = saved.get("cuda_rng")

    if (cuda_rng is not None) != (device.type == "cuda"):
        raise ValueError(
            "乱数状態を含む再開はCPU→CPUまたはCUDA→CUDAで行ってください"
        )

    if cuda_rng is not None and len(cuda_rng) != 1:
        raise ValueError(
            "この再開処理は1 GPUのチェックポイントを対象にしています"
        )

    model.load_state_dict(saved["model"])

    ema.net.load_state_dict(saved["ema"])
    ema.num_updates = saved["ema_num_updates"]

    ema_099.net.load_state_dict(saved["ema_099"])
    ema_099.num_updates = saved["ema_099_num_updates"]

    opt.load_state_dict(saved["optimizer"])

    # モデル構築などの乱数消費が終わってから復元する。
    feature_rng.set_state(saved["feature_rng"])
    torch.set_rng_state(saved["cpu_rng"])

    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng[0], device)