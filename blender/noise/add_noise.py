# Add DIC displacement noise to synthetic images
# Applies Gaussian blur using MatchID-derived noise floor

import argparse
import glob
import os
import numpy as np
import tifffile


# MatchID noise floor values (avg per-frame spatial std dev of u,v in pixels)
NOISE_FLOOR = {
    "undeformed": {"sigma_u": 0.0136, "sigma_v": 0.0124},
    "deformed":   {"sigma_u": 0.0149, "sigma_v": 0.0111},
}


def impose_noise(synthetic_image, sigma_u, sigma_v):
    """
    Apply anisotropic Gaussian blur using DIC-derived noise floor as sigma.
    sigma_u: horizontal blur (pixels)
    sigma_v: vertical blur (pixels)
    """
    from scipy.ndimage import gaussian_filter
    max_val = np.iinfo(synthetic_image.dtype).max
    blurred = gaussian_filter(synthetic_image.astype(float), sigma=(sigma_v, sigma_u))
    return np.clip(np.round(blurred), 0, max_val).astype(synthetic_image.dtype)


def main():
    parser = argparse.ArgumentParser(
        description="Add DIC displacement noise to synthetic images."
    )
    parser.add_argument("dataset", choices=["undeformed", "deformed"],
                        help="Which MatchID noise floor to use")
    parser.add_argument("input_dir", help="Directory containing synthetic TIFF images")
    parser.add_argument("output_dir", help="Directory to save noisy images")
    args = parser.parse_args()

    sigma_u = NOISE_FLOOR[args.dataset]["sigma_u"]
    sigma_v = NOISE_FLOOR[args.dataset]["sigma_v"]
    print(f"Using MatchID {args.dataset} noise floor:")
    print(f"  σ_u = {sigma_u:.4f} px")
    print(f"  σ_v = {sigma_v:.4f} px")

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.tif")) +
                   glob.glob(os.path.join(args.input_dir, "*.tiff")))
    print(f"\nFound {len(files)} images in {args.input_dir}")

    for f in files:
        img = tifffile.imread(f)
        noisy = impose_noise(img, sigma_u, sigma_v)
        out_path = os.path.join(args.output_dir, os.path.basename(f))
        tifffile.imwrite(out_path, noisy, compression='zlib')

    print(f"Saved {len(files)} noisy images to {args.output_dir}/")


if __name__ == "__main__":
    main()
