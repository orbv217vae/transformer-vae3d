"""Export a trusted training checkpoint to exact float32 inference tensors."""
import argparse
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from vae3d.model import VAE3DConfig, ViTVAE3D


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--expected-step', required=True, type=int)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / 'model.safetensors'
    partial = args.output / '.model.safetensors.partial'
    if any((args.output / name).exists() for name in ['model.safetensors', '.model.safetensors.partial', 'config.json', 'weights_info.json']):
        raise FileExistsError('Export destination contains artifacts; choose an empty directory')
    payload = torch.load(args.source, map_location='cpu', mmap=True, weights_only=True)
    if payload['step'] != args.expected_step:
        raise ValueError('Checkpoint step differs from expected step')
    saved = payload['model_config']
    if saved.get('hierarchical_spatial', False):
        raise ValueError('This release supports the non-hierarchical architecture only')
    names = {f.name for f in fields(VAE3DConfig)}
    allowed_extra = {'hierarchical_spatial', 'spatial_channels', 'spatial_decoder_min_channels', 'spatial_res_blocks', 'activation_checkpointing'}
    if set(saved) - names - allowed_extra:
        raise ValueError('Unknown checkpoint architecture fields')
    cfg = VAE3DConfig(**{k: v for k, v in saved.items() if k in names})
    sd = payload['model']
    if not isinstance(sd, dict) or not all(isinstance(v, torch.Tensor) and v.dtype == torch.float32 for v in sd.values()):
        raise ValueError('Expected a float32 model state dictionary')
    with torch.device('meta'):
        model = ViTVAE3D(cfg)
    model.load_state_dict(sd, strict=True, assign=True)
    parameter_count = sum(v.numel() for v in model.parameters())
    if parameter_count != 525766400:
        raise ValueError(f'Unexpected parameter count: {parameter_count}')
    print('validated architecture; exporting model tensors only', flush=True)
    save_file({k: v.detach().contiguous() for k, v in sd.items()}, str(partial))
    with safe_open(str(partial), framework='pt', device='cpu') as f:
        if set(f.keys()) != set(sd):
            raise ValueError('Export key mismatch')
        for k, v in sd.items():
            if not torch.equal(f.get_tensor(k), v):
                raise ValueError(f'Export tensor mismatch: {k}')
    partial.replace(target)
    (args.output / 'config.json').write_text(json.dumps(asdict(cfg), indent=2) + '\n')
    info = {'format': 'safetensors', 'dtype': 'float32', 'parameters': parameter_count,
            'tensor_count': len(sd), 'size_bytes': target.stat().st_size,
            'sha256': sha256(target),
            'all_tensors_equal_to_source': True,
            'contains_optimizer_or_discriminator': False}
    (args.output / 'weights_info.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info, indent=2), flush=True)


if __name__ == '__main__':
    main()
