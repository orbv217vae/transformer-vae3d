# Release validation

The released model bundle contains 168 float32 tensors with 525,766,400 model parameters. All exported tensors were checked for exact equality with the source checkpoint. No optimizer or discriminator state is stored in the safetensors file.

- Exported model: 2,103,083,096 bytes (approximately 2.10 GB).
- Weight SHA-256: `99e028cebaf328624191089d662fdf0af930943c1f18215f5be4c90c60cbd60e`.

The standalone implementation was compared with the original implementation on an RTX A6000 using PyTorch 2.5.1, float32 weights, bf16 autocast, and TF32 disabled. Only synthetic inputs were used.

| Input shape (D,H,W) | Latent shape (B,C,D,H,W) | Maximum absolute output difference |
|---|---|---:|
| [32, 40, 48] | [1, 16, 4, 5, 6] | 0.0 |
| [128, 128, 128] | [1, 16, 16, 16, 16] | 0.0 |
| [17, 25, 33] | [1, 16, 3, 4, 5] | 0.0 |

Posterior means, log-variances, low-level decoder outputs, and padded/cropped reconstruction outputs were bitwise identical in all three cases. These checks establish implementation compatibility, not reconstruction quality on unseen clinical data.

Six synthetic tests passed, covering strict safetensors loading, weight locators, latent geometry, non-divisible input padding, CT boundary padding, NIfTI physical geometry, end-to-end CLI reconstruction, and refusal to overwrite outputs. Synthetic CT preprocessing also matched the evaluated preprocessing pipeline exactly. Normalized inputs are clamped to [0,1] after float32 arithmetic, matching the evaluated model adapter's input clamp.
