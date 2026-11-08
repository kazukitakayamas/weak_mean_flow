import math
import torch
from .time_sampling import bc


def adaptive_error(sq, cfg):
    return sq / (sq.detach() + cfg["norm_eps"]) ** cfg["norm_p"]


def draw_comparison_features(d, cfg, generator):
    device = generator.device
    m = cfg["features"]

    normal = lambda *shape: torch.randn(
        *shape,
        device=device,
        generator=generator,
        dtype=torch.float64,
    )

    return (
        normal(m, d) * (cfg["spatial_scale"] / math.sqrt(d)),
        normal(m) * cfg["time_scale"],
        normal(m) * cfg["time_scale"],
        torch.rand(m, device=device, generator=generator, dtype=torch.float64) * (2 * math.pi))


def weak_summary(model, z, y, t, r, features, cfg):
    # ネットワークはfloat32、弱モーメントの集計はfloat64。
    f = (bc(t - r) * model(z, r, t, aug_cond=None)).flatten(1).double()

    zz = z.flatten(1).double()
    yy = y.flatten(1).double()
    tt = t.double()[:, None]
    rr = r.double()[:, None]

    wz, wr, wt, phase = features

    xi = zz @ wz.T + rr * wr + tt * wt + phase
    phi = (1 - tt) * xi.cos()

    # 従来の Aφ = ∂tφ + y・∇zφ
    aphi = (-xi.cos() - (1 - tt) * xi.sin() * (yy @ wz.T + wt))

    # logit-normalの時刻分布に対応する補正。
    # score = ∂t log q(r,t), q(r,t) = 2 p(r) p(t)
    logit_t = tt.log() - torch.log1p(-tt)
    score = (2 * tt - 1 - (logit_t - cfg["time_mean"]) / cfg["time_std"] ** 2) / (tt * (1 - tt))

    aphi = aphi + score * phi

    # s = Σ_i K_i  （形状 M × d）
    s = aphi.T @ f + phi.T @ yy

    # q = Σ_i ||K_i||²
    # K_iを(B,M,d)で保持せず、展開式で計算する。
    q = (
        aphi.square() * f.square().sum(1)[:, None]
        + phi.square() * yy.square().sum(1)[:, None]
        + 2 * aphi * phi * (f * yy).sum(1)[:, None]
    ).sum()

    return s, q



def backward_comparison(model, method, batch, cfg, feature_rng):
    assert method == "weak"

    z, y, t, r = batch
    device = z.device

    b, d = len(z), z[0].numel()
    nd, no = b // 2, b - b // 2
    mb = cfg["micro_batch"]

    # objective、対角MSE、弱U統計量
    metrics = torch.zeros(3, device=device, dtype=torch.float64)

    # ① 添付版と同じ適応重み付き対角損失
    for start in range(0, nd, mb):

        sl = slice(start, min(start + mb, nd))

        error = model(z[sl], t[sl], t[sl], aug_cond=None) - y[sl]

        sq = error.flatten(1).square().sum(1)
        loss = adaptive_error(sq, cfg).sum() / b
        loss.backward()

        metrics[0] += loss.detach()
        metrics[1] += (sq.detach().double().sum() / (nd * d))

    # ② 非対角512組全体の弱損失
    # 全サンプル・両方の計算で同じテスト関数を使用する。
    features = draw_comparison_features(d, cfg, feature_rng)

    s_all = torch.zeros(cfg["features"], d, device=device, dtype=torch.float64)
    q_all = torch.zeros((), device=device, dtype=torch.float64)

    # 1回目：勾配を保持せず、全非対角サンプルを集計。
    with torch.no_grad():

        for start in range(nd, b, mb):

            sl = slice(start, min(start + mb, b))

            s, q = weak_summary(model, z[sl], y[sl], t[sl], r[sl], features, cfg)

            s_all += s
            q_all += q

    denominator = no * (no - 1) * cfg["features"] * d

    weak_value = (s_all.square().sum() - q_all) / denominator

    # 添付版の非対角配分1/2を維持し、重み20を掛ける。
    factor = (no / b) * cfg["lambda_off"]

    # 2回目：全非対角サンプルのU統計量と同じ勾配を計算。
    for start in range(nd, b, mb):
      
        sl = slice(start, min(start + mb, b))

        s, q = weak_summary(model, z[sl], y[sl], t[sl], r[sl], features, cfg)

        surrogate = (2 * (s_all * s).sum() - q) / denominator

        (factor * surrogate).backward()

    # 表示には本来のU統計量を使う。
    metrics[0] += factor * weak_value
    metrics[2] = weak_value

    return metrics
