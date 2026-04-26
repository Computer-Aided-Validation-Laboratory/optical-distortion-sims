"""
Compute and compare epipolar error between undeformed and deformed DIC data.

Estimates the fundamental matrix from the undeformed reference frame,
then evaluates the epipolar error across all frames in both datasets.
"""

import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__)
UNDEF_DIR = os.path.join(BASE_DIR, "processed_data", "experimental_undeformed")
DEF_DIR = os.path.join(BASE_DIR, "processed_data", "experimental_deformed")
FIG_DIR = os.path.join(BASE_DIR, "figures")

# Camera intrinsics (for undistortion)
CAM1 = {"Cx": 2654.43, "Cy": 2302.98, "Fx": 19558.8, "Fy": 19558.8,
        "Kappa1": 0.00381827, "Kappa2": -0.0298443}
CAM2 = {"Cx": 2678.98, "Cy": 2304.68, "Fx": 19569.1, "Fy": 19569.1,
        "Kappa1": 0.00288466, "Kappa2": -0.0215253}


def undistort_points(x, y, cam):
    """Undistort pixel coordinates using radial distortion (Brown's model)."""
    xn = (x - cam["Cx"]) / cam["Fx"]
    yn = (y - cam["Cy"]) / cam["Fy"]
    r2 = xn**2 + yn**2
    radial = 1 + cam["Kappa1"] * r2 + cam["Kappa2"] * r2**2
    return (cam["Cx"] + (x - cam["Cx"]) * radial,
            cam["Cy"] + (y - cam["Cy"]) * radial)


def estimate_fundamental(p1, p2, max_pts=5000):
    """Estimate F using the normalised 8-point algorithm (subsampled)."""
    n = len(p1)
    if n > max_pts:
        idx = np.random.default_rng(42).choice(n, max_pts, replace=False)
        p1s, p2s = p1[idx], p2[idx]
    else:
        p1s, p2s = p1, p2

    def normalise(pts):
        mean = pts[:, :2].mean(axis=0)
        std = pts[:, :2].std()
        T = np.array([[1/std, 0, -mean[0]/std],
                      [0, 1/std, -mean[1]/std],
                      [0, 0, 1]])
        return (T @ pts.T).T, T

    p1n, T1 = normalise(p1s)
    p2n, T2 = normalise(p2s)

    A = np.column_stack([
        p2n[:, 0] * p1n[:, 0], p2n[:, 0] * p1n[:, 1], p2n[:, 0],
        p2n[:, 1] * p1n[:, 0], p2n[:, 1] * p1n[:, 1], p2n[:, 1],
        p1n[:, 0], p1n[:, 1], np.ones(len(p1n)),
    ])

    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Enforce rank-2
    U, S, Vt2 = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vt2

    # Denormalise
    F = T2.T @ F @ T1
    return F / np.linalg.norm(F)


def epipolar_error(F, p1, p2):
    """Point-to-epipolar-line distance in pixels."""
    l = F @ p1.T  # (3, N)
    numerator = np.abs(np.sum(p2 * l.T, axis=1))
    denominator = np.sqrt(l[0]**2 + l[1]**2)
    return numerator / denominator


def load_csv(filepath):
    df = pd.read_csv(filepath, skipinitialspace=True)
    df.columns = df.columns.str.strip().str.strip('"')
    return df


def extract_stereo_points(df):
    """Extract undistorted stereo point pairs from a DIC CSV, filtering sigma==-1."""
    valid = df["sigma"].values != -1
    x1 = df["x"].values[valid].astype(float)
    y1 = df["y"].values[valid].astype(float)
    q = df["q"].values[valid].astype(float)
    r = df["r"].values[valid].astype(float)
    x2 = x1 + q
    y2 = y1 + r

    x1u, y1u = undistort_points(x1, y1, CAM1)
    x2u, y2u = undistort_points(x2, y2, CAM2)

    ones = np.ones_like(x1u)
    p1 = np.column_stack([x1u, y1u, ones])
    p2 = np.column_stack([x2u, y2u, ones])
    return p1, p2, x1, y1, valid


def process_dataset(data_dir, F):
    """Compute per-frame epipolar error statistics for a dataset."""
    pattern = os.path.join(data_dir, "Image_*_0.csv")
    files = sorted(glob.glob(pattern))
    print(f"  Found {len(files)} frames in {os.path.basename(data_dir)}")

    frame_means = []
    frame_medians = []
    frame_stds = []
    all_errors = []

    for f in files:
        df = load_csv(f)
        p1, p2, _, _, _ = extract_stereo_points(df)
        errors = epipolar_error(F, p1, p2)
        frame_means.append(errors.mean())
        frame_medians.append(np.median(errors))
        frame_stds.append(errors.std())
        all_errors.append(errors)

    return {
        "means": np.array(frame_means),
        "medians": np.array(frame_medians),
        "stds": np.array(frame_stds),
        "all_errors": all_errors,
        "files": files,
    }


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    # ── Estimate F independently for each dataset ──
    print("Estimating fundamental matrix from undeformed data...")
    undef_ref_path = os.path.join(UNDEF_DIR, "Image_0001_0.csv")
    df_undef_ref = load_csv(undef_ref_path)
    p1_undef, p2_undef, x1_ref, y1_ref, valid_ref = extract_stereo_points(df_undef_ref)
    F_undef = estimate_fundamental(p1_undef, p2_undef)
    ref_errors = epipolar_error(F_undef, p1_undef, p2_undef)
    print(f"  Reference frame error: mean={ref_errors.mean():.4f} px")

    print("Estimating fundamental matrix from deformed data...")
    def_ref_path = os.path.join(DEF_DIR, "Image_0001_0.csv")
    df_def_ref = load_csv(def_ref_path)
    p1_def, p2_def, _, _, _ = extract_stereo_points(df_def_ref)
    F_def = estimate_fundamental(p1_def, p2_def)
    def_ref_errors = epipolar_error(F_def, p1_def, p2_def)
    print(f"  Reference frame error: mean={def_ref_errors.mean():.4f} px")

    # ── Process each dataset with its own F ──
    print("\nProcessing datasets...")
    undef_results = process_dataset(UNDEF_DIR, F_undef)
    def_results = process_dataset(DEF_DIR, F_def)

    # ── Print summary ──
    print(f"\n{'='*60}")
    print(f"{'Dataset':<20} {'Mean (px)':<12} {'Median (px)':<14} {'Std (px)':<12}")
    print(f"{'='*60}")
    print(f"{'Undeformed':<20} {undef_results['means'].mean():<12.4f} "
          f"{undef_results['medians'].mean():<14.4f} {undef_results['stds'].mean():<12.4f}")
    print(f"{'Deformed':<20} {def_results['means'].mean():<12.4f} "
          f"{def_results['medians'].mean():<14.4f} {def_results['stds'].mean():<12.4f}")
    print(f"{'='*60}")

    ratio = def_results['means'].mean() / undef_results['means'].mean()
    print(f"\nDeformed/Undeformed ratio: {ratio:.2f}x")

    # ── Plot 1: Mean epipolar error vs frame number ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(undef_results["means"], label="Undeformed", marker=".", markersize=3)
    ax.plot(def_results["means"], label="Deformed", marker=".", markersize=3)
    ax.set_xlabel("Frame number")
    ax.set_ylabel("Mean epipolar error (pixels)")
    ax.set_title("Mean Epipolar Error: Undeformed vs Deformed")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "epipolar_error_comparison.png"), dpi=150)
    plt.close(fig)

    # ── Plot 2: Histograms side by side ──
    all_undef = np.concatenate(undef_results["all_errors"])
    all_def = np.concatenate(def_results["all_errors"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    ax1.hist(all_undef, bins=100, edgecolor="black", linewidth=0.3, color="steelblue")
    ax1.set_xlabel("Epipolar Error (pixels)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Undeformed (mean={all_undef.mean():.4f} px)")
    ax1.axvline(all_undef.mean(), color="r", linestyle="--", label=f"Mean")
    ax1.legend()

    ax2.hist(all_def, bins=100, edgecolor="black", linewidth=0.3, color="coral")
    ax2.set_xlabel("Epipolar Error (pixels)")
    ax2.set_title(f"Deformed (mean={all_def.mean():.4f} px)")
    ax2.axvline(all_def.mean(), color="r", linestyle="--", label=f"Mean")
    ax2.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "epipolar_error_histograms.png"), dpi=150)
    plt.close(fig)

    # ── Plot 3: Spatial map for one deformed frame (last frame) ──
    df_last = load_csv(def_results["files"][-1])
    p1_last, p2_last, x1_last, y1_last, _ = extract_stereo_points(df_last)
    errors_last = epipolar_error(F_def, p1_last, p2_last)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    # Undeformed reference
    sc1 = ax1.scatter(x1_ref, y1_ref, c=ref_errors, s=0.5, cmap="hot",
                      vmin=0, vmax=max(ref_errors.max(), errors_last.max()))
    ax1.set_title("Undeformed (ref frame)")
    ax1.set_xlabel("x (pixels)")
    ax1.set_ylabel("y (pixels)")
    ax1.set_aspect("equal")
    ax1.invert_yaxis()

    # Deformed last frame
    sc2 = ax2.scatter(x1_last, y1_last, c=errors_last, s=0.5, cmap="hot",
                      vmin=0, vmax=max(ref_errors.max(), errors_last.max()))
    ax2.set_title(f"Deformed (last frame)")
    ax2.set_xlabel("x (pixels)")
    ax2.set_aspect("equal")
    ax2.invert_yaxis()

    plt.colorbar(sc2, ax=[ax1, ax2], label="Epipolar error (px)", shrink=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "epipolar_error_spatial_comparison.png"), dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
