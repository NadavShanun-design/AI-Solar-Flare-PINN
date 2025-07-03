"""
preprocess_magnetograms.py
-------------------------
Loads, preprocesses, and batches SDO/HMI vector magnetogram data for use in JAX-based models.

Dependencies:
    pip install sunpy numpy scipy astropy

Usage:
    python preprocess_magnetograms.py --input_dir <path> --output_file <path>

Outputs:
    - Preprocessed magnetogram batches as NumPy arrays (JAX compatible)
"""
import os
import argparse
import numpy as np
from glob import glob
from astropy.io import fits
from scipy.ndimage import gaussian_filter

# Optional: import sunpy for advanced data handling
try:
    import sunpy.map
except ImportError:
    sunpy = None

def load_magnetogram_fits(file_path):
    """Load a vector magnetogram FITS file and return Bx, By, Bz arrays."""
    with fits.open(file_path) as hdul:
        # Assumes Bx, By, Bz are in extensions 1, 2, 3 (common for HMI SHARP)
        bx = hdul[1].data.astype(np.float32)
        by = hdul[2].data.astype(np.float32)
        bz = hdul[3].data.astype(np.float32)
    return bx, by, bz

def preprocess_magnetogram(bx, by, bz, norm_mode='per_image', smooth_sigma=0, mask_threshold=None):
    """Normalize, optionally smooth, and mask the magnetogram components."""
    # Stack for joint processing
    mag = np.stack([bx, by, bz], axis=0)  # shape: (3, H, W)
    # Normalization
    if norm_mode == 'per_image':
        mean = mag.mean(axis=(1,2), keepdims=True)
        std = mag.std(axis=(1,2), keepdims=True) + 1e-6
        mag = (mag - mean) / std
    elif norm_mode == 'global':
        mean = mag.mean()
        std = mag.std() + 1e-6
        mag = (mag - mean) / std
    # Smoothing
    if smooth_sigma > 0:
        mag = np.array([gaussian_filter(m, sigma=smooth_sigma) for m in mag])
    # Masking
    if mask_threshold is not None:
        mask = np.sqrt(np.sum(mag**2, axis=0)) > mask_threshold
        mag = mag * mask[None, :, :]
    return mag

def batch_magnetograms(mag_list, batch_size):
    """Batch a list of magnetograms into arrays of shape (N, 3, H, W)."""
    batches = [np.stack(mag_list[i:i+batch_size]) for i in range(0, len(mag_list), batch_size)]
    return batches

def main(args):
    files = sorted(glob(os.path.join(args.input_dir, '*.fits')))
    print(f"Found {len(files)} FITS files.")
    mags = []
    for f in files:
        bx, by, bz = load_magnetogram_fits(f)
        mag = preprocess_magnetogram(
            bx, by, bz,
            norm_mode=args.norm_mode,
            smooth_sigma=args.smooth_sigma,
            mask_threshold=args.mask_threshold
        )
        mags.append(mag)
    batches = batch_magnetograms(mags, args.batch_size)
    # Save as .npz for JAX/NumPy loading
    np.savez_compressed(args.output_file, batches=batches)
    print(f"Saved {len(batches)} batches to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess SDO/HMI vector magnetograms for JAX models.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with FITS files')
    parser.add_argument('--output_file', type=str, required=True, help='Output .npz file')
    parser.add_argument('--norm_mode', type=str, default='per_image', choices=['per_image', 'global'], help='Normalization mode')
    parser.add_argument('--smooth_sigma', type=float, default=0.0, help='Gaussian smoothing sigma')
    parser.add_argument('--mask_threshold', type=float, default=None, help='Mask threshold for field strength')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    args = parser.parse_args()
    main(args) 