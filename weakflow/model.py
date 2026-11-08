import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(shape, mode, fan_in, fan_out):
    if mode == "xavier_uniform":
        return math.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
        
    raise ValueError(f"この U-Net では使わない初期化: {mode}")


class Linear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, init_mode="xavier_uniform", init_weight=1, init_bias=0):
        super().__init__()
        kw = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = nn.Parameter(weight_init([out_features, in_features], **kw) * init_weight)
        self.bias = nn.Parameter(weight_init([out_features], **kw) * init_bias) if bias else None   # init_bias=0 なので 0（乱数は使う）

    def forward(self, x):
        x = x @ self.weight.t()

        return x + self.bias if self.bias is not None else x


class Conv2d(nn.Module):
    # 畳み込み（up / down のときは FIR フィルタで解像度を 2 倍 / 1/2 にしてから畳み込む）

    def __init__(self, in_channels, out_channels, kernel, bias=True, up=False, down=False, resample_filter=(1, 1), init_mode="xavier_uniform", init_weight=1, init_bias=0):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down

        kw = dict(mode=init_mode, fan_in=in_channels * kernel * kernel, fan_out=out_channels * kernel * kernel)

        self.weight = nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **kw) * init_weight) if kernel else None
        self.bias = nn.Parameter(weight_init([out_channels], **kw) * init_bias) if kernel and bias else None

        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = torch.outer(f, f)[None, None] / f.sum().square()             # [1,3,3,1] の外積、和 1

        self.register_buffer("resample_filter", f if up or down else None)

    def forward(self, x):

        w = self.weight
        b = self.bias
        f = self.resample_filter

        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.up:
            x = F.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)

        if self.down:
            x = F.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)

        if w is not None:
            x = F.conv2d(x, w, padding=w_pad)

        if b is not None:
            x = x + b.reshape(1, -1, 1, 1)

        return x


class GroupNorm(nn.Module):

    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()

        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):

        N, C = x.shape[:2]
        g = x.reshape(N, self.num_groups, C // self.num_groups, *x.shape[2:])   # (N, G, C/G, H, W)
        dims = tuple(range(2, g.dim()))
        mean = g.mean(dim=dims, keepdim=True)
        var = g.var(dim=dims, keepdim=True, unbiased=False)
        x = ((g - mean) / torch.sqrt(var + self.eps)).reshape(x.shape)

        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


def qkv_attention(qkv, n_heads):
    # 公式（Flow Matching リポジトリ由来）の QKVAttention：q, k の両方に ch^{-1/4} を掛けてから内積
    bs, width, length = qkv.shape
    ch = width // (3 * n_heads)
    q, k, v = qkv.chunk(3, dim=1)
    scale = 1 / math.sqrt(math.sqrt(ch))
    w = torch.einsum("bct,bcs->bts", (q * scale).reshape(bs * n_heads, ch, length), (k * scale).reshape(bs * n_heads, ch, length))
    w = torch.softmax(w.float(), dim=-1).type(w.dtype)
    a = torch.einsum("bts,bcs->bct", w, v.reshape(bs * n_heads, ch, length))

    return a.reshape(bs, -1, length)


class UNetBlock(nn.Module):
    # SongUNet の残差ブロック（adaptive_scale=False の場合だけ）

    def __init__(self, in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
                 num_heads=1, dropout=0.0, skip_scale=math.sqrt(0.5), eps=1e-6, resample_filter=(1, 3, 3, 1),
                 init=None, init_zero=None, init_attn=None):
        super().__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.num_heads = num_heads if attention else 0
        self.dropout, self.skip_scale = dropout, skip_scale

        self.norm0 = GroupNorm(in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels, out_channels, 3, up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(emb_channels, out_channels, **init)
        self.norm1 = GroupNorm(out_channels, eps=eps)
        self.conv1 = Conv2d(out_channels, out_channels, 3, **init_zero)
        self.skip = None

        if out_channels != in_channels or up or down:
            self.skip = Conv2d(in_channels, out_channels, 1, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(out_channels, eps=eps)
            self.qkv = Conv2d(out_channels, out_channels * 3, 1, **init_attn)
            self.proj = Conv2d(out_channels, out_channels, 1, **init_zero)


    def forward(self, x, emb):
        orig = x
        x = self.conv0(F.silu(self.norm0(x)))
        x = F.silu(self.norm1(x + self.affine(emb)[:, :, None, None]))          # 時間条件は足し算で入る
        x = self.conv1(F.dropout(x, p=self.dropout, training=self.training))
        x = (x + (self.skip(orig) if self.skip is not None else orig)) * self.skip_scale

        if self.num_heads:
            a = qkv_attention(self.qkv(self.norm2(x)).flatten(2), self.num_heads)
            x = (self.proj(a.reshape(x.shape)) + x) * self.skip_scale

        return x


def positional_embedding(s, num_channels, max_positions=10000, endpoint=True):

    # 時刻 s (B,) → [sin(s ω_j), cos(s ω_j)]  (B, num_channels)、ω_j = max_positions^{-j/(num_channels/2 - 1)}
    freqs = torch.arange(num_channels // 2, dtype=torch.float32, device=s.device)
    freqs = (1 / max_positions) ** (freqs / (num_channels // 2 - (1 if endpoint else 0)))
    a = torch.outer(s, freqs.to(s.dtype))

    return torch.cat([a.sin(), a.cos()], dim=1)   # 公式は [cos, sin] を作ってから flip で入れ替えている


class SongUNet(nn.Module):

    def __init__(self, img_resolution=32, in_channels=3, out_channels=3, augment_dim=0,
                 model_channels=128, channel_mult=(1, 2, 2, 2), channel_mult_emb=4, num_blocks=4,
                 attn_resolutions=(16,), dropout=0.10, channel_mult_noise=1, resample_filter=(1, 1)):
        super().__init__()

        emb_channels = model_channels * channel_mult_emb             # 512
        self.noise_channels = noise_channels = model_channels * channel_mult_noise   # 256
        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=math.sqrt(0.2))

        bk = dict(
            emb_channels=emb_channels, num_heads=1,
            dropout=dropout,
            skip_scale=math.sqrt(0.5),
            eps=1e-6,
            resample_filter=resample_filter,
            init=init, init_zero=init_zero,
            init_attn=init_attn
            )

        # 時間条件：[PE(t), PE(h)]（2·noise_channels 次元）＋ 拡張ラベル → MLP
        self.map_augment = Linear(augment_dim, noise_channels * 2, bias=False, **init) if augment_dim else None
        self.map_layer0 = Linear(noise_channels * 2, emb_channels, **init)
        self.map_layer1 = Linear(emb_channels, emb_channels, **init)

        # U-Net の下り
        self.enc = nn.ModuleDict()
        cout = in_channels

        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level

            if level == 0:
                cin, cout = cout, model_channels
                self.enc[f"{res}x{res}_conv"] = Conv2d(cin, cout, 3, **init)

            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock(cout, cout, down=True, **bk)

            for idx in range(num_blocks):

                cin, cout = cout, model_channels * mult
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(cin, cout, attention=(res in attn_resolutions), **bk)

        skips = [blk.out_channels for blk in self.enc.values()]

        # U-Net の上り
        self.dec = nn.ModuleDict()

        for level, mult in reversed(list(enumerate(channel_mult))):

            res = img_resolution >> level

            if level == len(channel_mult) - 1:
                self.dec[f"{res}x{res}_in0"] = UNetBlock(cout, cout, attention=True, **bk)
                self.dec[f"{res}x{res}_in1"] = UNetBlock(cout, cout, **bk)

            else:
                self.dec[f"{res}x{res}_up"] = UNetBlock(cout, cout, up=True, **bk)

            for idx in range(num_blocks + 1):

                cin, cout = cout + skips.pop(), model_channels * mult
                attn = idx == num_blocks and res in attn_resolutions

                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(cin, cout, attention=attn, **bk)

        self.dec[f"{img_resolution}x{img_resolution}_aux_norm"] = GroupNorm(cout, eps=1e-6)
        self.dec[f"{img_resolution}x{img_resolution}_aux_conv"] = Conv2d(cout, out_channels, 3, **init_zero)

    def forward(self, x, t, h, aug_cond=None):

        emb = torch.cat([positional_embedding(t, self.noise_channels), positional_embedding(h, self.noise_channels)], dim=1)

        if self.map_augment is not None and aug_cond is not None:
            emb = emb + self.map_augment(aug_cond)

        emb = F.silu(self.map_layer1(F.silu(self.map_layer0(emb))))

        skips = []
        for blk in self.enc.values():

            x = blk(x, emb) if isinstance(blk, UNetBlock) else blk(x)
            skips.append(x)

        for name, blk in self.dec.items():

            if name.endswith("aux_norm"):
                x = blk(x)

            elif name.endswith("aux_conv"):
                x = blk(F.silu(x))

            else:
                if "_block" in name:
                    x = torch.cat([x, skips.pop()], dim=1)   # skip を連結
                x = blk(x, emb)
        return x



class MeanFlowNet(SongUNet):
    # u_θ(z, r, t)：平均速度。ネットワークには (t, h = t − r) を渡す（公式の u_func と同じ）
    def forward(self, z, r, t, aug_cond=None):
        return super().forward(z, t, t - r, aug_cond)
