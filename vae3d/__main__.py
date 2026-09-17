import argparse
from pathlib import Path
import torch
from .inference import VAEInference, load_model
from .io import prepare_nifti, save_reconstruction


def main():
    p = argparse.ArgumentParser(description="Reconstruct one 3D NIfTI with the f8/C16 Transformer VAE")
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--modality", choices=["mri", "ct"], required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--output-scale", choices=["source", "normalized"], default="source")
    p.add_argument("--no-ct-padding-repair", action="store_true")
    args = p.parse_args()
    if args.output.exists() or args.input.resolve() == args.output.resolve():
        p.error("Output must be a new file, distinct from input")
    if not str(args.output).endswith((".nii", ".nii.gz")):
        p.error("Output must end in .nii or .nii.gz")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    prepared = prepare_nifti(args.input, args.modality, not args.no_ct_padding_repair)
    model = load_model(args.weights, args.config, args.device)
    runner = VAEInference(model, args.precision)
    try:
        recon = runner.reconstruct(prepared.array01)
    except torch.cuda.OutOfMemoryError:
        raise SystemExit("Full-volume inference exceeded GPU memory. Use a larger GPU; input was not cropped or tiled.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_reconstruction(recon, prepared, args.output, args.output_scale)
    print(f"Reconstructed {prepared.array01.shape} on the 1 mm grid; output scale={args.output_scale}")


if __name__ == "__main__":
    main()
