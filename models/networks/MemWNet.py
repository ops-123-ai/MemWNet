import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from utils import A_CDP, At_CDP

# 1 -> 32
class Head(nn.Module):
    def __init__(self, embed_dim=32, drop_path=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1, embed_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, padding=1),
        )
        self.alpha = nn.Parameter(1e-2 * torch.ones(1, embed_dim, 1, 1))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        return x + self.drop_path(self.alpha * self.block(x))

# 32 -> 1
class Tail(nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.block(x)

# channel_mixer
class ConvGLU(nn.Module):
    def __init__(self, dim, expansion=4):
        super().__init__()
        hid = dim * expansion
        self.norm   = nn.GroupNorm(1, dim)
        self.w1     = nn.Conv2d(dim, hid, 1, bias=False)
        self.w2     = nn.Conv2d(dim, hid, 1, bias=False)
        self.dwconv = nn.Conv2d(hid, hid, 3, 1, 1, groups=hid, bias=False)
        self.act    = nn.GELU()
        self.w3     = nn.Conv2d(hid, dim, 1, bias=False)

    def forward(self, x):
        xn   = self.norm(x)
        gate = self.act(self.dwconv(self.w2(xn)))
        return self.w3(self.w1(xn) * gate)

# GMB
class ImprovedTransformer(nn.Module):
    def __init__(self, dim, pool_size=3):
        super().__init__()
        self.norm1         = nn.GroupNorm(1, dim)
        self.token_mixer   = nn.AvgPool2d(pool_size, 1, pool_size // 2,
                                          count_include_pad=False)
        self.channel_mixer = ConvGLU(dim)

    def forward(self, x):
        x = x + self.token_mixer(self.norm1(x))
        x = x + self.channel_mixer(x)
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )
    def forward(self, x): return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )
    def forward(self, x): return self.body(x)

# unet skip
class Fuse(nn.Module):
    def __init__(self, n_feat): super().__init__()
    def forward(self, enc, dec): return enc + dec


# Haar DWT and IDWT
class HaarDWT(nn.Module):
    def forward(self, x):
        x00, x10 = x[:, :, 0::2, 0::2], x[:, :, 1::2, 0::2]
        x01, x11 = x[:, :, 0::2, 1::2], x[:, :, 1::2, 1::2]
        LL = (x00 + x10 + x01 + x11) * 0.5
        LH = (x00 + x10 - x01 - x11) * 0.5
        HL = (x00 - x10 + x01 - x11) * 0.5
        HH = (x00 - x10 - x01 + x11) * 0.5
        return LL, LH, HL, HH


class HaarIDWT(nn.Module):
    def forward(self, LL, LH, HL, HH):
        x00 = (LL + LH + HL + HH) * 0.5
        x10 = (LL + LH - HL - HH) * 0.5
        x01 = (LL - LH + HL - HH) * 0.5
        x11 = (LL - LH - HL + HH) * 0.5
        B, C, H, W = LL.shape
        res = torch.zeros(B, C, H*2, W*2, device=LL.device, dtype=LL.dtype)
        res[:, :, 0::2, 0::2] = x00;  res[:, :, 1::2, 0::2] = x10
        res[:, :, 0::2, 1::2] = x01;  res[:, :, 1::2, 1::2] = x11
        return res

# TWSB
class FPEB_SoftThreshold(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwt = HaarDWT()
        self.idwt = HaarIDWT()

        self.conv_in_l1  = nn.Conv2d(dim * 4, dim * 4, 1, bias=False)
        self.conv_out_l1 = nn.Conv2d(dim * 4, dim * 4, 1, bias=False)

        self.conv_in_l2  = nn.Conv2d(dim * 4, dim * 4, 1, bias=False)
        self.conv_out_l2 = nn.Conv2d(dim * 4, dim * 4, 1, bias=False)

        self.threshold_raw_l1 = nn.Parameter(torch.tensor([0.005]))
        self.threshold_raw_l2 = nn.Parameter(torch.tensor([0.003]))

    def _soft_thresh(self, x, raw_threshold):
        thr = torch.abs(raw_threshold)
        return torch.sign(x) * F.relu(torch.abs(x) - thr)

    def forward(self, x):
        c = x.shape[1]
        
        LL1, LH1, HL1, HH1 = self.dwt(x)
        freq_l1 = self.conv_in_l1(torch.cat([LL1, LH1, HL1, HH1], dim=1))
        LL1_feat = freq_l1[:, 0:c]
        HF1      = freq_l1[:, c:]

        LL2, LH2, HL2, HH2 = self.dwt(LL1_feat)

        freq_l2 = self.conv_in_l2(torch.cat([LL2, LH2, HL2, HH2], dim=1))

        LL2_feat = freq_l2[:, 0:c]     
        HF2_soft = self._soft_thresh(freq_l2[:, c:], self.threshold_raw_l2)
        HF1_soft = self._soft_thresh(HF1,             self.threshold_raw_l1)

        out_l2 = self.conv_out_l2(torch.cat([LL2_feat, HF2_soft], dim=1))
        LL1_recon = self.idwt(
            out_l2[:, 0:c],
            out_l2[:, c:2*c],
            out_l2[:, 2*c:3*c],
            out_l2[:, 3*c:4*c]
        )

        out_l1 = self.conv_out_l1(
            torch.cat([LL1_recon, HF1_soft], dim=1)
        )
        return self.idwt(
            out_l1[:, 0:c],
            out_l1[:, c:2*c],
            out_l1[:, 2*c:3*c],
            out_l1[:, 3*c:4*c]
        )

# MGUB
class UNET_core(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Encoder
        self.enc1   = ImprovedTransformer(dim)
        self.down1  = Downsample(dim)
        self.enc2   = ImprovedTransformer(dim * 2)
        self.down2  = Downsample(dim * 2)
        # Bottleneck
        self.latent = nn.Sequential(
            ImprovedTransformer(dim * 4),
            ConvGLU(dim * 4),
            ImprovedTransformer(dim * 4),
        )
        # Decoder
        self.up2    = Upsample(dim * 4)
        self.fuse2  = Fuse(dim * 2)
        self.dec2   = ImprovedTransformer(dim * 2)
        self.up1    = Upsample(dim * 2)
        self.fuse1  = Fuse(dim)
        self.dec1   = ImprovedTransformer(dim)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        l  = self.latent(self.down2(e2))
        d2 = self.dec2(self.fuse2(e2, self.up2(l)))
        return self.dec1(self.fuse1(e1, self.up1(d2)))

# CFMB
class DepthWiseAttnRes(nn.Module):
    def __init__(self, feat_dim: int, num_stages: int):
        super().__init__()

        self.queries = nn.ParameterList([
            nn.Parameter(torch.zeros(feat_dim))
            for _ in range(num_stages)
        ])

        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, feat_curr, hidden_history, stage_idx):

        # stage 0 -> residual
        if len(hidden_history) == 0:
            return feat_curr

        sources = [feat_curr] + hidden_history # N × [B,C,H,W]
        V = torch.stack(sources, dim=1) # [B, N, C, H, W]

        V_gap = V.mean(dim=[-2, -1]) # [B, N, C]
        K = F.normalize(V_gap, dim=-1, eps=1e-6) # [B, N, C]

        q = self.queries[stage_idx] # [C]

        logits = torch.einsum('c, bnc -> bn', q, K) # [B, N]
        logits = logits * self.temperature

        alpha = logits.softmax(dim=1) # [B, N]

        alpha_4d = alpha[:, :, None, None, None] # [B, N, 1, 1, 1]
        feat_agg = (alpha_4d * V).sum(dim=1) # [B, C, H, W]

        return feat_agg


# single stage
class ProxBlock(nn.Module):
    def __init__(self, feat_dim=32, num_stages=7):
        super().__init__()
        self.lambda_gd      = nn.Parameter(torch.tensor([0.5]))
        self.f_pre          = Head(feat_dim)
        self.attn_res       = DepthWiseAttnRes(feat_dim, num_stages)
        self.unet1          = UNET_core(feat_dim)
        self.soft_threshold = FPEB_SoftThreshold(feat_dim)
        self.unet2          = UNET_core(feat_dim)
        self.f_out          = Tail(feat_dim)

    def forward(self, x, b, rate, mask, stage_idx, hidden_history, device='cuda'):

        z = A_CDP(x, rate, mask, device=device)
        z_abs = torch.abs(z)
        resid = z - b * (z / (z_abs + 1e-8))
        grad = At_CDP(resid, rate, mask)
        r = x - self.lambda_gd.abs() * grad

        feat = self.f_pre(r)
        feat_agg = self.attn_res(feat, hidden_history, stage_idx)

        e1 = self.unet1.enc1(feat_agg)
        e2 = self.unet1.enc2(self.unet1.down1(e1))
        l  = self.unet1.latent(self.unet1.down2(e2))
        d2 = self.unet1.dec2(self.unet1.fuse2(e2, self.unet1.up2(l)))
        u1 = self.unet1.dec1(self.unet1.fuse1(e1, self.unet1.up1(d2)))

        soft = self.soft_threshold(u1)
        u2   = self.unet2(soft)
        out  = self.f_out(u2)
        return r + out, u2, r


class unfold_net(nn.Module):
    def __init__(self, num_stages=7, feat_dim=32, alpha_init=0.9, beta_init=0.8):
        super().__init__()
        self.num_stages = num_stages

        self.stages = nn.ModuleList([ProxBlock(feat_dim, num_stages) for _ in range(num_stages)])

        self.alphas = nn.ParameterList([nn.Parameter(torch.tensor(alpha_init)) for _ in range(num_stages)])
        self.betas = nn.ParameterList([nn.Parameter(torch.tensor(beta_init)) for _ in range(num_stages)])

    def forward(self, x, b, rate, mask, device='cuda'):
        hidden_history = []

        x_km2 = x.clone()
        x_km1 = x.clone()
        x_k   = x.clone()

        for k, stage in enumerate(self.stages):
            alpha = self.alphas[k]
            beta  = self.betas[k]

            if k == 0:
                x_prox, u2, _ = stage(x_k, b, rate, mask, stage_idx=0, hidden_history=hidden_history, device=device)
                x_km2 = x_k
                x_km1 = x_prox
                x_k   = x_prox

            else:
                x_prox, u2, _ = stage(x_km1, b, rate, mask, stage_idx=k, hidden_history=hidden_history, device=device)
                x_next = (1 - alpha) * x_km2 + (alpha - beta) * x_km1 + beta * x_prox
                x_km2 = x_km1
                x_km1 = x_next
                x_k   = x_next

            hidden_history.append(u2)

        return x_k