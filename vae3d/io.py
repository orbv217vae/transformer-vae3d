"""NIfTI preprocessing with ANTs geometry retained on the output 1 mm grid."""
from dataclasses import dataclass
import numpy as np
from scipy import ndimage


@dataclass
class PreparedVolume:
    array01: np.ndarray
    reference: object
    low: float
    high: float
    padding_fraction: float


def repair_ct_padding(values, sentinel=-3024.0, air=-1000.0):
    """Replace only exact sentinel voxels connected to any array boundary."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 3 or not np.isfinite(arr).all():
        raise ValueError("Expected a finite 3D HU array")
    candidate = arr == sentinel
    seeds = np.zeros(arr.shape, dtype=bool)
    for axis in range(3):
        for edge in (0, -1):
            sl = [slice(None)] * 3
            sl[axis] = edge
            seeds[tuple(sl)] = candidate[tuple(sl)]
    padding = ndimage.binary_propagation(seeds, mask=candidate)
    result = arr.copy()
    result[padding] = air
    return result, float(padding.mean())


def prepare_nifti(path, modality="mri", ct_padding=True):
    import ants
    if modality not in ("mri", "ct"):
        raise ValueError("modality must be mri or ct")
    image = ants.image_read(str(path))
    if image.dimension != 3:
        raise ValueError("Expected a single 3D volume; split MRI sequences before inference")
    values = image.numpy().astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Input contains nonfinite intensities")
    fraction = 0.0
    if modality == "ct" and ct_padding:
        values, fraction = repair_ct_padding(values)
    image = ants.from_numpy(values, origin=image.origin, spacing=image.spacing, direction=image.direction)
    image = ants.reorient_image2(image, "RAS")
    # ANTs code 1 is nearest-neighbor, retained to match evaluated preprocessing.
    image = ants.resample_image(image, (1., 1., 1.), use_voxels=False, interp_type=1)
    arr = np.ascontiguousarray(image.numpy().transpose(2, 1, 0), dtype=np.float32)
    low, high = np.percentile(arr, [0.5, 99.5])
    if not high > low:
        raise ValueError("Degenerate percentile range")
    np.clip(arr, low, high, out=arr)
    arr -= low
    arr /= high - low + 1e-8
    # Float32 percentile arithmetic can overshoot an endpoint by one ULP.
    # The evaluated model adapter also clamps before mapping to [-1,1].
    np.clip(arr, 0.0, 1.0, out=arr)
    return PreparedVolume(arr, image, float(low), float(high), fraction)


def save_reconstruction(array01, prepared, output, scale="source"):
    import ants
    arr = np.asarray(array01, dtype=np.float32)
    if arr.shape != prepared.array01.shape or not np.isfinite(arr).all():
        raise ValueError("Reconstruction must be finite and match the prepared grid")
    if scale not in ("source", "normalized"):
        raise ValueError("scale must be source or normalized")
    if scale == "source":
        arr = arr * (prepared.high - prepared.low) + prepared.low
    ref = prepared.reference
    image = ants.from_numpy(np.ascontiguousarray(arr.transpose(2, 1, 0)),
                            origin=ref.origin, spacing=ref.spacing, direction=ref.direction)
    ants.image_write(image, str(output))
