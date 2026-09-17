"""Safe weight loading and whole-volume inference with explicit latent geometry."""
from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
import torch
import torch.nn.functional as F

from .model import VAE3DConfig, ViTVAE3D


def load_model(weights, config, device="cpu"):
    weights = Path(weights)
    if weights.suffix == '.path':
        locator = weights.read_text().strip()
        if not locator or '\n' in locator:
            raise ValueError("Weight locator must contain exactly one path")
        located = Path(locator).expanduser()
        weights = located if located.is_absolute() else weights.parent / located
    if not weights.is_file():
        raise FileNotFoundError(weights)
    cfg = json.loads(Path(config).read_text())
    cfg['patch_size'] = tuple(cfg['patch_size'])
    # Construct on CPU: nonpersistent RoPE buffers must be materialized too.
    model = ViTVAE3D(VAE3DConfig(**cfg))
    model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
    return model.eval().requires_grad_(False).to(device)


@dataclass(frozen=True)
class VolumeGeometry:
    original_shape: tuple
    padded_shape: tuple
    pad_before: tuple


def pad_volume(x, patch):
    if x.ndim != 5 or any(s < 1 for s in x.shape):
        raise ValueError("Expected nonempty (B,C,D,H,W) input")
    shape = tuple(x.shape[-3:])
    total = tuple((-s) % p for s, p in zip(shape, patch))
    before = tuple(t // 2 for t in total)
    pads = tuple(v for b, t in reversed(list(zip(before, total))) for v in (b, t - b))
    padded = F.pad(x, pads, value=0.0)
    return padded, VolumeGeometry(shape, tuple(padded.shape[-3:]), before)


class VAEInference:
    def __init__(self, model, precision="bf16"):
        if model.cfg.in_channels != 1 or model.cfg.out_channels != 1 or model.cfg.out_activation != "tanh":
            raise ValueError("Volume inference requires a single-channel model with tanh output")
        if precision not in ("fp32", "bf16", "fp16"):
            raise ValueError("precision must be fp32, bf16, or fp16")
        self.model = model.eval().requires_grad_(False)
        self.device = next(model.parameters()).device
        self.precision = precision
        if self.device.type != "cuda" and precision != "fp32":
            raise ValueError("CPU inference requires precision='fp32'")
        if precision == "bf16" and self.device.type == "cuda":
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError("GPU does not support bf16; choose fp32 or fp16")

    def _autocast(self):
        if self.precision == "fp32":
            return nullcontext()
        return torch.autocast("cuda", dtype={'bf16': torch.bfloat16, 'fp16': torch.float16}[self.precision])

    @torch.inference_mode()
    def encode(self, image01, sample=False):
        """Encode (B,1,D,H,W) values in [0,1] to raw (B,16,Dz,Hy,Wx) latents.

        No latent scaling or shift is applied. Sampling is opt-in; mode is default.
        Return (latents, geometry); retain geometry to remove symmetric padding.
        """
        x = torch.as_tensor(image01, dtype=torch.float32, device=self.device)
        if x.ndim != 5 or x.shape[1] != self.model.cfg.in_channels:
            raise ValueError("Expected (B,1,D,H,W) input")
        if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            raise ValueError("Input must be finite and normalized to [0,1]")
        work, geometry = pad_volume(x, self.model.cfg.patch_size)
        with self._autocast():
            encoded = self.model.encode(work.mul(2).sub(1))
            z = encoded.latent_dist.sample() if sample else encoded.latent_dist.mode()
        z = z.transpose(1, 2).reshape(x.shape[0], self.model.cfg.latent_dim, *encoded.grid_shape)
        return z, geometry

    @torch.inference_mode()
    def decode(self, latents, geometry=None):
        """Decode raw 5D Gaussian latents to float32 [0,1] images.

        With no geometry, return the full latent-grid-times-eight image.
        """
        z = torch.as_tensor(latents, device=self.device, dtype=torch.float32)
        if z.ndim != 5 or z.shape[1] != self.model.cfg.latent_dim or any(n < 1 for n in z.shape):
            raise ValueError("Expected nonempty (B,C,Dz,Hy,Wx) latent tensor")
        if not torch.isfinite(z).all():
            raise ValueError("Latents must be finite")
        grid = tuple(z.shape[-3:])
        target = tuple(g * p for g, p in zip(grid, self.model.cfg.patch_size))
        if geometry is not None:
            if target != geometry.padded_shape or len(geometry.original_shape) != 3 or len(geometry.pad_before) != 3:
                raise ValueError("Geometry does not match latent grid")
            if any(s <= 0 or b < 0 or b + s > t for s, b, t in zip(geometry.original_shape, geometry.pad_before, target)):
                raise ValueError("Invalid crop geometry")
        tokens = z.flatten(2).transpose(1, 2).contiguous()
        with self._autocast():
            result = self.model.decode(tokens, grid)
        result = result.float().add(1).mul(0.5).clamp(0, 1)
        if geometry is not None:
            crop = tuple(slice(b, b + s) for b, s in zip(geometry.pad_before, geometry.original_shape))
            result = result[(slice(None), slice(None), *crop)]
        return result

    def reconstruct(self, array01):
        array = np.asarray(array01, dtype=np.float32)
        if array.ndim != 3:
            raise ValueError("Expected one (D,H,W) volume")
        z, geometry = self.encode(array[None, None])
        result = self.decode(z, geometry)[0, 0].cpu().numpy()
        if not np.isfinite(result).all():
            raise RuntimeError("Nonfinite reconstruction; retry with fp32 precision")
        return result
