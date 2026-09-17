"""Inference backbone. Tensor operations retain the trained model conventions."""

from dataclasses import dataclass

from typing import Optional, Tuple

import torch

from torch import nn

import torch.nn.functional as F

class DiagonalGaussian:
    def __init__(self, mu: torch.Tensor, logvar: torch.Tensor):
        self.mu = mu
        self.logvar = logvar

    @torch.no_grad()
    def mode(self):
        return self.mu

    def sample(self):
        std = (0.5 * self.logvar).exp()
        eps = torch.randn_like(std)
        return self.mu + eps * std

    def kl(self):
        return -0.5 * (1 + self.logvar - self.mu.pow(2) - self.logvar.exp())

@dataclass
class EncodeOutput:
    latent_dist: DiagonalGaussian
    grid_shape: Tuple[int, int, int]   # (Dz, Dy, Dx)

def exists(x):
    return x is not None

class RoPE3D(nn.Module):
    def __init__(self, head_dim: int, base_theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        rope_dim = (head_dim // 6) * 6
        self.rope_dim = rope_dim
        self.base_theta = base_theta

        if rope_dim > 0:
            twoL = rope_dim // 3
            twoL = (twoL // 2) * 2
            self.twoL = twoL
            self.rope_dim = twoL * 3

            d_pair = twoL // 2
            inv_freq = 1.0 / (base_theta ** (torch.arange(d_pair).bfloat16() / d_pair))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        else:
            self.twoL = 0
            self.register_buffer("inv_freq", torch.tensor([]), persistent=False)

    def _build_angles(self, npos: int, device):
        d_pair = self.inv_freq.shape[0]
        t = torch.arange(npos, device=device).bfloat16().unsqueeze(-1)       # (npos,1)
        return t * self.inv_freq.unsqueeze(0)                             # (npos,d_pair)

    @staticmethod
    def _apply_rotary_pairs(x_axis_chunk, cos, sin):
        B, Hn, N, twoL = x_axis_chunk.shape
        d_pair = twoL // 2

        x_pair = x_axis_chunk.view(B, Hn, N, d_pair, 2)
        x_even = x_pair[..., 0]
        x_odd  = x_pair[..., 1]

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_even * sin + x_odd * cos

        x_rot = torch.stack((x_rot_even, x_rot_odd), dim=-1).reshape(B, Hn, N, twoL)
        return x_rot

    def forward(self, q, k, D: int, H: int, W: int):
        if self.rope_dim == 0:
            return q, k

        B, Hn, N, Hd = q.shape
        device = q.device
        twoL = self.twoL
        rope_dim = self.rope_dim

        q_rot, q_rest = q[..., :rope_dim], q[..., rope_dim:]
        k_rot, k_rest = k[..., :rope_dim], k[..., rope_dim:]

        qx, qy, qz = torch.split(q_rot, [twoL, twoL, twoL], dim=-1)
        kx, ky, kz = torch.split(k_rot, [twoL, twoL, twoL], dim=-1)

        idx = torch.arange(N, device=device)
        zz = idx // (H * W)
        yy = (idx // W) % H
        xx = idx % W

        d_pair = twoL // 2
        angles_x = self._build_angles(W, device)            # (W, d_pair)
        cos_x = angles_x.cos()[xx]                          # (N, d_pair)
        sin_x = angles_x.sin()[xx]
        angles_y = self._build_angles(H, device)            # (H, d_pair)
        cos_y = angles_y.cos()[yy]
        sin_y = angles_y.sin()[yy]
        angles_z = self._build_angles(D, device)            # (D, d_pair)
        cos_z = angles_z.cos()[zz]
        sin_z = angles_z.sin()[zz]

        qx = self._apply_rotary_pairs(qx, cos_x, sin_x)
        qy = self._apply_rotary_pairs(qy, cos_y, sin_y)
        qz = self._apply_rotary_pairs(qz, cos_z, sin_z)

        kx = self._apply_rotary_pairs(kx, cos_x, sin_x)
        ky = self._apply_rotary_pairs(ky, cos_y, sin_y)
        kz = self._apply_rotary_pairs(kz, cos_z, sin_z)

        q_out = torch.cat([qx, qy, qz, q_rest], dim=-1)
        k_out = torch.cat([kx, ky, kz, k_rest], dim=-1)
        return q_out, k_out

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * norm

class SwiGLU(nn.Module):
    # https://arxiv.org/abs/2002.05202 + SiLU gating
    def __init__(self, dim: int, mult: float = 4.0):
        super().__init__()
        inner = int(dim * mult)
        self.proj = nn.Linear(dim, inner * 2, bias=False)
        self.out = nn.Linear(inner, dim, bias=False)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return self.out(F.silu(gate) * x)

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        rope: Optional[RoPE3D] = None,
        layer_scale_init: float = 1e-4,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        inner = heads * dim_head

        self.qkv = nn.Linear(dim, inner * 3, bias=qkv_bias)
        self.proj = nn.Linear(inner, dim, bias=False)
        self.attn_dropout = attn_dropout
        self.resid_dropout = resid_dropout

        self.rope = rope
        self.gamma = nn.Parameter(torch.ones(dim) * layer_scale_init)

    def forward(self, x, D: int, H: int, W: int):
        B, N, C = x.shape
        qkv = self.qkv(x)  # (B,N,3*inner)
        q, k, v = qkv.chunk(3, dim=-1)
        Hn, Dh = self.heads, self.dim_head

        q = q.view(B, N, Hn, Dh).transpose(1, 2)  # (B,H,N,Dh)
        k = k.view(B, N, Hn, Dh).transpose(1, 2)
        v = v.view(B, N, Hn, Dh).transpose(1, 2)

        if exists(self.rope):
            q, k = self.rope(q, k, D, H, W)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )  # (B,H,N,Dh)

        out = attn_out.transpose(1, 2).contiguous().view(B, N, Hn * Dh)
        out = self.proj(out)
        return x + out * self.gamma

class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_mult: float,
        rope: Optional[RoPE3D],
        layer_scale_init: float = 1e-4,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(
            dim, heads, dim_head, rope,
            layer_scale_init=layer_scale_init,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
        )
        self.norm2 = RMSNorm(dim)
        self.mlp = nn.Sequential(
            SwiGLU(dim, mult=mlp_mult),
        )
        self.gamma2 = nn.Parameter(torch.ones(dim) * layer_scale_init)

    def forward(self, x, D, H, W):
        x = x.to(x.dtype)
        x = self.attn(self.norm1(x), D, H, W)
        x = x + self.mlp(self.norm2(x)) * self.gamma2
        return x

class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels: int, dim: int, patch_size: Tuple[int, int, int]):
        super().__init__()
        pz, py, px = patch_size
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, dim, kernel_size=(pz, py, px), stride=(pz, py, px))

    def forward(self, x):
        # x: (B,C,D,H,W) -> (B, N, dim), with N = D'*H'*W'
        _, _, Din, Hin, Win = x.shape
        pz, py, px = self.patch_size
        if (Din % pz) != 0 or (Hin % py) != 0 or (Win % px) != 0:
            raise ValueError(
                f"Input spatial shape {(Din, Hin, Win)} must be divisible by patch_size {self.patch_size}."
            )
        x = self.proj(x)  # (B,dim,D',H',W')
        B, C, D, H, W = x.shape
        x = x.view(B, C, D * H * W).transpose(1, 2).contiguous()
        return x, (D, H, W)

class PatchUnembed3D(nn.Module):
    def __init__(self, out_channels: int, dim: int, patch_size: Tuple[int, int, int], out_activation: Optional[str] = None):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.out_activation = out_activation
        pz, py, px = patch_size
        self.patch_proj = nn.Linear(dim, out_channels * pz * py * px, bias=False)

    def forward(self, tokens, grid_shape, target_shape):
        """
        tokens: (B,N,dim)
        grid_shape: (D',H',W')
        target_shape: (D,H,W)
        """
        B, N, C = tokens.shape
        Dg, Hg, Wg = grid_shape
        pz, py, px = self.patch_size
        D, H, W = target_shape

        patches = self.patch_proj(tokens)  # (B,N,out*C*pz*py*px)
        patches = patches.view(B, N, self.out_channels, pz, py, px)
        patches = patches.view(B, Dg, Hg, Wg, self.out_channels, pz, py, px)
        patches = patches.permute(0, 4, 1, 5, 2, 6, 3, 7)  # (B,C,Dg,pz,Hg,py,Wg,px)
        out = patches.reshape(B, self.out_channels, Dg * pz, Hg * py, Wg * px)

        if exists(self.out_activation):
            if self.out_activation == "sigmoid":
                out = out.sigmoid()
            elif self.out_activation == "tanh":
                out = out.tanh()
        return out

@dataclass
class VAE3DConfig:
    in_channels: int = 1
    out_channels: int = 1
    patch_size: Tuple[int, int, int] = (8, 8, 8)
    dim: int = 1280
    depth_enc: int = 10
    depth_dec: int = 10
    heads: int = 16
    dim_head: int = 80
    mlp_mult: float = 4.0
    latent_dim: int = 16
    rope_base_theta: float = 10000.0
    out_activation: Optional[str] = "tanh"


class ViTEncoder3D(nn.Module):
    def __init__(self, cfg, rope):
        super().__init__()
        self.patch = PatchEmbed3D(cfg.in_channels, cfg.dim, cfg.patch_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.dim, cfg.heads, cfg.dim_head, cfg.mlp_mult, rope)
            for _ in range(cfg.depth_enc)
        ])
        self.norm = RMSNorm(cfg.dim)

    def forward(self, x):
        tokens, grid = self.patch(x)
        for block in self.blocks:
            tokens = block(tokens, *grid)
        return self.norm(tokens), grid


class ViTDecoder3D(nn.Module):
    def __init__(self, cfg, rope):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.dim, cfg.heads, cfg.dim_head, cfg.mlp_mult, rope)
            for _ in range(cfg.depth_dec)
        ])
        self.norm = RMSNorm(cfg.dim)
        self.unpatch = PatchUnembed3D(cfg.out_channels, cfg.dim, cfg.patch_size, cfg.out_activation)

    def forward(self, tokens, grid, target):
        for block in self.blocks:
            tokens = block(tokens, *grid)
        return self.unpatch(self.norm(tokens), grid, target)


class ViTVAE3D(nn.Module):
    """One-channel VAE. Core API uses [-1,1] images and (B,N,C) latent tokens.

    Pass grid explicitly to decode; no mutable last-input geometry is retained.
    RoPE arithmetic intentionally retains the checkpoint's bfloat16 convention.
    """
    def __init__(self, cfg: VAE3DConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.heads * cfg.dim_head != cfg.dim:
            raise ValueError("heads * dim_head must equal dim")
        if len(cfg.patch_size) != 3 or any(p <= 0 for p in cfg.patch_size):
            raise ValueError("patch_size must contain three positive integers")
        rope = RoPE3D(cfg.dim_head, cfg.rope_base_theta)
        self.encoder = ViTEncoder3D(cfg, rope)
        self.to_mu = nn.Linear(cfg.dim, cfg.latent_dim, bias=False)
        self.to_logvar = nn.Linear(cfg.dim, cfg.latent_dim, bias=False)
        self.from_z = nn.Linear(cfg.latent_dim, cfg.dim, bias=False)
        self.decoder = ViTDecoder3D(cfg, rope)

    def encode(self, x):
        tokens, grid = self.encoder(x)
        return EncodeOutput(DiagonalGaussian(self.to_mu(tokens), self.to_logvar(tokens)), grid)

    def decode(self, z, grid):
        if z.ndim != 3 or len(grid) != 3 or any(int(g) != g or g < 1 for g in grid):
            raise ValueError("Expected (B,N,C) tokens and a positive integer 3D grid")
        grid = tuple(int(g) for g in grid)
        if z.shape[1] != grid[0] * grid[1] * grid[2] or z.shape[2] != self.cfg.latent_dim:
            raise ValueError("Latent token shape does not match grid or channel count")
        target = tuple(g * p for g, p in zip(grid, self.cfg.patch_size))
        return self.decoder(self.from_z(z), grid, target)

    def forward(self, x):
        posterior = self.encode(x)
        return self.decode(posterior.latent_dist.mode(), posterior.grid_shape)
