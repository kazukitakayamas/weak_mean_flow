import torch

from .data import sample_data


def bc(s):
    return s.view(-1, 1, 1, 1)


def interpolate(x, e, t):
    return (1 - bc(t)) * x + bc(t) * e


def sample_imf_times(n, cfg, device):
    pair = torch.sigmoid(
        torch.randn(n, 2, device=device) * cfg["time_std"]
        + cfg["time_mean"]
    )

    t = pair.max(dim=1).values
    r = pair.min(dim=1).values

    # 弱損失内でlog(t)、log(1-t)を使うため確認する。
    if not bool(((pair > 0) & (pair < 1)).all()):
        raise FloatingPointError("時刻が0または1に丸められました")

    # 前半が対角、後半が非対角。
    r[:n // 2] = t[:n // 2]

    return t, r


def draw_comparison_batch(cfg, x_all, device):
    x = sample_data(x_all, cfg["batch_size"], device)

    flip = torch.rand(len(x), 1, 1, 1, device=device) < 0.5
    x = torch.where(flip, x.flip(-1), x)

    e = torch.randn_like(x)
    t, r = sample_imf_times(len(x), cfg, device)

    return interpolate(x, e, t), e - x, t, r