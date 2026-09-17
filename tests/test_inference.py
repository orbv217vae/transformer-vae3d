from dataclasses import asdict
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from vae3d import VAE3DConfig, ViTVAE3D, VAEInference, VolumeGeometry, load_model
from vae3d.io import repair_ct_padding, prepare_nifti, save_reconstruction


@pytest.fixture
def weights_dir(tmp_path):
    base = os.environ.get('VAE_TEST_WEIGHT_DIR')
    if base:
        with tempfile.TemporaryDirectory(prefix='vae_test_', dir=base) as directory:
            yield Path(directory)
    else:
        yield tmp_path


@pytest.fixture
def runner():
    torch.manual_seed(4)
    torch.set_num_threads(2)
    cfg = VAE3DConfig(dim=48, depth_enc=1, depth_dec=1, heads=2, dim_head=24, latent_dim=4)
    return VAEInference(ViTVAE3D(cfg), precision='fp32')


def test_latent_layout_and_padding_match_direct_model(runner):
    x = torch.rand(1, 1, 17, 26, 33)
    z, geometry = runner.encode(x)
    assert z.shape == (1, 4, 3, 4, 5)
    assert geometry.pad_before == (3, 3, 3)
    padded = torch.nn.functional.pad(x, (3, 4, 3, 3, 3, 4))
    posterior = runner.model.encode(padded * 2 - 1)
    assert torch.equal(z.flatten(2).transpose(1, 2), posterior.latent_dist.mode())
    expected = ((runner.model.decode(posterior.latent_dist.mode(), posterior.grid_shape).float() + 1) / 2).clamp(0, 1)
    result = runner.decode(z, geometry)
    torch.testing.assert_close(result, expected[:, :, 3:20, 3:29, 3:36], rtol=0, atol=0)
    torch.testing.assert_close(runner.decode(z), expected, rtol=0, atol=0)
    assert torch.equal(result, runner.decode(*runner.encode(x)))


def test_decode_without_prior_encode_and_bad_geometry(runner):
    z = torch.zeros(1, 4, 2, 3, 4)
    assert runner.decode(z).shape == (1, 1, 16, 24, 32)
    with pytest.raises(ValueError):
        runner.decode(z, VolumeGeometry((16, 24, 32), (24, 24, 32), (0, 0, 0)))
    with pytest.raises(ValueError):
        runner.model.decode(torch.zeros(1, 4, 4), (2, 3, 4))
    with pytest.raises(ValueError):
        runner.encode(torch.full((1, 1, 8, 8, 8), 2.0))


def test_weights_load_strictly_without_training_state(tmp_path, weights_dir, runner):
    w = weights_dir / 'model.safetensors'
    c = tmp_path / 'config.json'
    save_file(runner.model.state_dict(), str(w))
    c.write_text(json.dumps(asdict(runner.model.cfg)))
    loaded = load_model(w, c)
    x = torch.rand(1, 1, 8, 16, 24) * 2 - 1
    torch.testing.assert_close(loaded(x), runner.model(x), rtol=0, atol=0)
    locator = tmp_path / 'weights.path'
    locator.write_text(os.path.relpath(w, tmp_path) + '\n')
    torch.testing.assert_close(load_model(locator, c)(x), loaded(x), rtol=0, atol=0)
    bad = runner.model.state_dict();bad.pop('to_mu.weight')
    save_file(bad, str(w))
    with pytest.raises(RuntimeError):
        load_model(w, c)


def test_ct_padding_exact_boundary_only():
    x = np.full((9, 10, 11), -1000, dtype=np.float32)
    x[0, :, :] = -3024
    x[1, 0, 0] = -3024
    x[4, 4, 4] = -3024
    x[5, 5, 5] = -1024
    original = x.copy()
    repaired, fraction = repair_ct_padding(x)
    np.testing.assert_array_equal(original, x)
    assert repaired[0, 2, 2] == -1000
    assert repaired[4, 4, 4] == -3024 and repaired[5, 5, 5] == -1024
    assert fraction == 111 / x.size


def test_nifti_roundtrip_retains_physical_geometry(tmp_path):
    import ants
    xyz = np.arange(15 * 17 * 19, dtype=np.float32).reshape(15, 17, 19)
    direction = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    original = ants.from_numpy(xyz, origin=(31., -19., 7.), spacing=(1.3, 1.7, 2.1), direction=direction)
    path, output = tmp_path / 'synthetic.nii.gz', tmp_path / 'recon.nii.gz'
    ants.image_write(original, str(path))
    prepared = prepare_nifti(path)
    save_reconstruction(prepared.array01, prepared, output, scale='normalized')
    restored = ants.image_read(str(output))
    np.testing.assert_allclose(restored.numpy(), prepared.array01.transpose(2, 1, 0), atol=1e-6)
    np.testing.assert_allclose(restored.origin, prepared.reference.origin, atol=1e-5)
    np.testing.assert_allclose(restored.spacing, prepared.reference.spacing, atol=1e-6)
    np.testing.assert_allclose(restored.direction, prepared.reference.direction, atol=1e-6)
    assert restored.shape == prepared.reference.shape
    expected = ants.resample_image(ants.reorient_image2(ants.image_read(str(path)), 'RAS'), (1., 1., 1.), use_voxels=False, interp_type=1)
    lo, hi = np.percentile(expected.numpy(), [.5, 99.5])
    expected01 = expected.numpy().transpose(2, 1, 0).copy()
    np.clip(expected01, lo, hi, out=expected01)
    expected01 -= lo
    expected01 /= hi-lo+1e-8
    np.clip(expected01, 0, 1, out=expected01)
    assert prepared.array01.dtype == np.float32
    np.testing.assert_array_equal(prepared.array01, expected01)


def test_cli_reconstructs_and_refuses_overwrite(tmp_path, weights_dir, runner):
    import ants
    import os
    import subprocess
    import sys
    w, c = weights_dir / 'model.safetensors', tmp_path / 'config.json'
    save_file(runner.model.state_dict(), str(w))
    c.write_text(json.dumps(asdict(runner.model.cfg)))
    source, output = tmp_path / 'input.nii.gz', tmp_path / 'output.nii.gz'
    image = ants.from_numpy(np.arange(17*19*21, dtype=np.float32).reshape(17,19,21), origin=(7.,9.,11.))
    ants.image_write(image, str(source))
    cmd = [sys.executable, '-B', '-m', 'vae3d', '--weights', str(w), '--config', str(c),
           '--input', str(source), '--output', str(output), '--modality', 'mri',
           '--device', 'cpu', '--precision', 'fp32', '--output-scale', 'normalized']
    env = {**os.environ, 'OMP_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2', 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS':'2'}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    prepared = prepare_nifti(source)
    actual = ants.image_read(str(output))
    expected = runner.reconstruct(prepared.array01)
    np.testing.assert_allclose(actual.numpy().transpose(2,1,0), expected, atol=1e-6)
    np.testing.assert_allclose(actual.origin, prepared.reference.origin, atol=1e-6)
    before = output.read_bytes()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert result.returncode != 0 and 'Output must be a new file' in result.stderr
    assert output.read_bytes() == before
