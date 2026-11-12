import math
import torch
from .time_sampling import bc, off_time_score


def adaptive_error(sq, cfg):
    return sq / (sq.detach() + cfg["norm_eps"]) ** cfg["norm_p"]


def color_edge_directions(m, d, generator):

    side, window = 32, 8
    if d != 3 * side * side or m < 2 or m % 2:
        raise ValueError("color/edge features require RGB 32x32 and an even count >= 2")
    
    half = m // 2
    device = generator.device
    rng = dict(device=device, generator=generator)

    # 各関数が見る8×8領域を、32×32画像の中から選ぶ
    axis = torch.arange(side, device=device)
    top, left = torch.randint(side - window + 1, (2, m, 1, 1), **rng)
    row = axis[None, :, None] - top
    col = axis[None, None, :] - left
    patch = (row >= 0) & (row < window) & (col >= 0) & (col < window)

    # 前半64本：領域内の R-G、G-B、B-R のいずれかを測る
    palette = torch.tensor(
        [[1, -1, 0], [0, 1, -1], [-1, 0, 1]],
        dtype = torch.float64,
        device = device,
    )
    channels = palette[torch.randint(3, (half,), **rng)]
    color = channels[:, :, None, None] * patch[:half, None]

    vertical = torch.randint(2, (half, 1, 1), **rng).bool()
    positive = torch.where(
        vertical,
        col[half:] < window // 2,
        row[half:] < window // 2,
    )

    signed_patch = patch[half:] * (2 * positive.to(torch.float64) - 1)
    edge = signed_patch[:, None].expand(-1, 3, -1, -1)

    wz = torch.cat((color, edge), dim=0).flatten(1)
    return torch.nn.functional.normalize(wz, dim=1)

def draw_comparison_features(d, cfg, generator):
    device = generator.device
    m = cfg["features"]

    normal = lambda *shape: torch.randn(
        *shape,
        device=device,
        generator=generator,
        dtype=torch.float64,
    )

    if "color_loss_share" in cfg:
        share = float(cfg["color_loss_share"])
        if not 0 <= share <= 1:
            raise ValueError("color_loss_share must be in [0, 1]")
        wz = color_edge_directions(m, d, generator) * cfg["spatial_scale"]

        amplitude = wz.new_empty(m)
        amplitude[:m // 2] = math.sqrt(2 * share) #色
        amplitude[m // 2:] = math.sqrt(2 * (1 - share)) #輪郭
    else:
        wz = normal(m, d) * (cfg["spatial_scale"] / math.sqrt(d))
        amplitude = 1.0

    return (
        wz,
        normal(m) * cfg["time_scale"],
        normal(m) * cfg["time_scale"],
        torch.rand(m, device=device, generator=generator, dtype=torch.float64) * (2 * math.pi),
        amplitude,
    )


def weak_summary(model, z, y, t, r, features, cfg):
    # ネットワークはfloat32、弱モーメントの集計はfloat64。
    f = (bc(t - r) * model(z, r, t, aug_cond=None)).flatten(1).double()

    zz = z.flatten(1).double()
    yy = y.flatten(1).double()
    tt = t.double()[:, None]
    rr = r.double()[:, None]

    wz, wr, wt, phase, amplitude = features

    xi = zz @ wz.T + rr * wr + tt * wt + phase

    cos_xi = amplitude * xi.cos()
    sin_xi = amplitude * xi.sin()
    
    phi = (1 - tt) * xi.cos_xi

    # 従来の Aφ = ∂tφ + y・∇zφ
    aphi = -cos_xi - (1 - tt) * sin_xi * (yy @ wz.T + wt)
    aphi = aphi + off_time_score(rr, tt, cfg) * phi

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



def backward_comparison(model, batch, cfg, feature_rng):

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
