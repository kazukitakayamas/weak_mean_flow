import math
import torch

from .data import sample_data


def bc(s):
    return s.view(-1, 1, 1, 1)


def interpolate(x, e, t):
    return (1 - bc(t)) * x + bc(t) * e


def off_time_score(r, t, cfg):
    """∂t log q(r,t) for sorted pairs; r,t are FP64 and 0<r<=t<1."""
    mu, sigma = cfg["time_mean"], cfg["time_std"]
    rho = cfg.get("off_uniform_prob", 0.0)
    logit_t = t.log() - torch.log1p(-t)
    base = (2*t - 1 - (logit_t - mu)/sigma**2) / (t*(1-t))
    if rho == 0:
        return base
    if rho == 1:
        return torch.zeros_like(base)

    def log_density(s):
        log_s, log_1ms = s.log(), torch.log1p(-s)
        normal = (log_s - log_1ms - mu) / sigma
        return -0.5*normal.square() - math.log(sigma*math.sqrt(2*math.pi)) - log_s - log_1ms

    # The factor 2 from sorting cancels between both mixture components.
    log_odds = math.log1p(-rho) - math.log(rho) + log_density(r) + log_density(t)
    return torch.sigmoid(log_odds) * base


def sample_imf_times(n, cfg, device, mix_rng=None):
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

    rho = cfg.get("off_uniform_prob", 0.0)
    if rho:
        off = slice(n // 2, n)
        count = n - n // 2
        use_uniform = torch.rand(count, device=device, generator=mix_rng) < rho
        pair_u = torch.rand(count, 2, device=device, generator=mix_rng)
        # torch.rand can return exactly 0; keep the inputs to log finite.
        pair_u.clamp_min_(torch.finfo(pair_u.dtype).tiny)
        r[off] = torch.where(use_uniform, pair_u.amin(dim=1), r[off])
        t[off] = torch.where(use_uniform, pair_u.amax(dim=1), t[off])

    return t, r


def draw_comparison_batch(cfg, x_all, device, mix_rng=None):
    x = sample_data(x_all, cfg["batch_size"], device)

    flip = torch.rand(len(x), 1, 1, 1, device=device) < 0.5
    x = torch.where(flip, x.flip(-1), x)

    e = torch.randn_like(x)
    t, r = sample_imf_times(len(x), cfg, device, mix_rng)

    return interpolate(x, e, t), e - x, t, r
