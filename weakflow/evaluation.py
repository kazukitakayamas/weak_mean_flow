import torch

from .time_sampling import bc


@torch.no_grad()
def generate(model, e, N=1, return_path=False):
    if N < 1:
        raise ValueError("NFEは1以上にしてください")

    model.eval()
    z = e.clone()

    # 軌跡が不要なら途中画像を保持しない。
    path = [z.clone()] if return_path else None
    grid = torch.linspace(1, 0, N + 1).tolist()

    for i in range(N):
        t = torch.full((len(z),), grid[i], device=z.device)
        r = torch.full((len(z),), grid[i + 1], device=z.device)

        z = z - bc(t - r) * model(z, r, t)

        if return_path:
            path.append(z.clone())

    return (z, torch.stack(path, 1)) if return_path else z


@torch.no_grad()
def compute_fid(net, x_all, N=1, n=10000, bs=64, seed=42):
    from torchmetrics.image.fid import FrechetInceptionDistance

    if n < 2 or bs < 1:
        raise ValueError("n >= 2、bs >= 1が必要です")

    device = next(net.parameters()).device
    net.eval()

    fid = FrechetInceptionDistance(normalize=True).to(device)

    # 実画像：学習データ50,000枚、拡張なし。
    for i in range(0, len(x_all), bs):
        fid.update(
            x_all[i:i + bs].to(device).float() / 255.0,
            real=True,
        )

    # 評価専用の乱数生成器を使う。
    generator = torch.Generator(device=device).manual_seed(seed)

    for i in range(0, n, bs):
        e = torch.randn(
            min(bs, n - i), 3, 32, 32,
            device=device,
            generator=generator,
        )

        xh = (generate(net, e, N) * 0.5 + 0.5).clamp(0, 1)

        fid.update(
            torch.floor(xh * 255) / 255.0,
            real=False,
        )

    return float(fid.compute())