# Transformer VAE for 3D medical images

Inference code for **A High-Compression Transformer VAE for Efficient 3D Medical Image Generation**.

The model uses equal 8× downsampling along each spatial axis and 16 Gaussian latent channels. It has 525,766,400 parameters. The release package prepared here contains the **CT-adapted step-4000, non-EMA** weights. This is a different checkpoint from the brain MRI step-8000 EMA model discussed in the paper. Training data are not distributed.

## Installation

Use Python 3.10 or newer and install a CUDA-compatible PyTorch build for your GPU, then:

```bash
python -m pip install -e '.[nifti]'
```

The tensor API requires only PyTorch, NumPy, and safetensors. The NIfTI command additionally uses ANTsPy and SciPy. The validated environment uses PyTorch 2.5.1, NumPy 2.2.6, safetensors 0.7.0, ANTsPy 0.5.4, and SciPy 1.14.1. There are no automatic model downloads or dependencies on the training project.

## Model files

Download the **CT-adapted step-4000 (non-EMA)** weight bundle from [Hugging Face: orbv217vae/transformer-vae3d-ct](https://huggingface.co/orbv217vae/transformer-vae3d-ct). The model repository is currently private and requires access. Keep the downloaded bundle in your Hugging Face cache and supply its absolute path to the inference commands. It contains:

- `model.safetensors`: float32 model tensors only; no optimizer, discriminator, or training state.
- `config.json`: architecture fields needed to construct the model.
- `weights_info.json`: model size, checkpoint variant, SHA-256, and export verification.

The same architecture configuration is included in `configs/ct_step4000.json`. Model weights are hosted separately on Hugging Face; they are not included in this Git repository. You can also pass a `.path` text file to `--weights`; its single line points to a separately stored safetensors file. Relative paths are resolved from the locator file directory. Local weight locators remain ignored by Git.

## Reconstruct a NIfTI volume

```bash
python -m vae3d \
  --weights weights/ct-step4000/model.safetensors \
  --config configs/ct_step4000.json \
  --input input.nii.gz \
  --output outputs/reconstruction.nii.gz \
  --modality ct --device cuda:0 --precision bf16
```

For MRI, use `--modality mri`. Each input is one 3D, single-channel volume; process different MRI sequences separately. CPU execution requires `--device cpu --precision fp32` and can be slow. `--precision fp32` is also available on CUDA. Existing output files are not overwritten.

Preprocessing follows the evaluated pipeline: ANTs RAS reorientation, 1 mm resampling, and whole-volume 0.5/99.5 percentile clipping and normalization to [0,1]. The existing resampling setting is **ANTs interpolation code 1 (nearest neighbor)**; it is retained deliberately for numerical compatibility. Array axes are transposed from ANTs XYZ to model DHW. The VAE receives [-1,1] values, with symmetric zero padding in [0,1] space to a multiple of eight. Reconstruction uses the posterior mean and removes that padding. It does not silently crop or tile the volume.

For CT, exact -3024 HU values connected to an image-array boundary are replaced with -1000 HU **before resampling**. Isolated interior values are preserved. CT inputs must already be in HU. Use `--no-ct-padding-repair` to disable this specific padding convention. No generic low-intensity mask or anatomical segmentation is applied.

The output is on the **preprocessed 1 mm grid**, retaining its origin, spacing, and direction. It is not on the original acquisition grid. By default, intensities are mapped back to the clipped source intensity range (HU for CT); values removed by percentile clipping cannot be recovered. Use `--output-scale normalized` for [0,1] output. Source images are never overwritten.

## Tensor and latent API

```python
import torch
from vae3d import load_model, VAEInference

model = load_model('weights/ct-step4000/model.safetensors',
                   'configs/ct_step4000.json', device='cuda:0')
runner = VAEInference(model, precision='bf16')
image01 = torch.rand(1, 1, 128, 128, 128, device='cuda:0')
z, geometry = runner.encode(image01)  # (1, 16, 16, 16, 16)
reconstruction01 = runner.decode(z, geometry)  # (1, 1, 128, 128, 128)
```

`encode` uses the posterior mean by default; `sample=True` samples from the diagonal Gaussian. Latents have **no implicit scaling factor or shift**. Downstream generators must establish their own latent normalization. Decode externally generated 5D latents with `runner.decode(z)`; the result is latent-grid size multiplied by the patch size. Retain `geometry` when recovering an original shape that required padding.

The low-level `model.encode` accepts [-1,1] input dimensions divisible by eight and returns `(B,N,16)` posterior tokens plus a spatial grid. `model.decode(tokens, grid)` requires an explicit grid. Low-level `forward` reconstructs from the posterior mean. The inference-only model preserves the trained layer names, attention computation, and bfloat16 RoPE convention. No training loop, FSDP, discriminator, or activation-checkpointing dependency is included.

Spatial-grid reduction is 512×; single-channel C16 scalar-element reduction is 32×. Neither quantity is a compressed file bitrate. Variable shapes are supported within available memory. Full-volume inference can still run out of memory for large inputs.

## Validation

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

Tests use synthetic inputs only and cover geometry, padding, output alignment, safe loading, and latent contracts. The prepared CT weight bundle is also compared tensor by tensor with its source checkpoint; a separate release audit records full-model equivalence against the original inference implementation.

## Exporting your own trusted checkpoint

```bash
python scripts/export_weights.py \
  --source training_checkpoint.pt \
  --expected-step 4000 \
  --output weights/ct-step4000
```

This exporter selects the regular `model` state, not EMA. It preserves float32 values exactly and refuses to overwrite an existing export. The original training checkpoint remains unchanged. The exported architecture must match this non-hierarchical f8/C16 model.

## Release status

This repository is an inference release candidate. No code or model license has been selected yet; license terms must be supplied by the owner before public distribution. Third-party dependencies retain their respective licenses. No patient images, private data lists, training logs, or credentials are included.
