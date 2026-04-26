"""
Compute the 3D affine transformation matrix between undeformed and deformed
experimental DIC displacement fields from MatchID.

The transformation captures the systematic change in measured displacement
introduced by the deformation of the glass window, which alters the optical
path between specimen and cameras.

Inputs:
    - Time-averaged displacement fields (u, v, w) for undeformed and deformed
      experimental datasets (100 frames each, exported from MatchID as CSVs).

Outputs:
    - 4x4 affine transformation matrix T mapping undeformed 3D positions to
      deformed 3D positions.
    - Split-validation residuals confirming the model is not overfitting.
"""

import numpy as np
import glob
import os


def load_displacement_stack(folder):
    """Load all per-frame displacement CSVs from a folder into a 3D array."""
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    stack = []
    for f in files:
        data = np.genfromtxt(f, delimiter=",")
        stack.append(data)
    min_rows = min(s.shape[0] for s in stack)
    min_cols = min(s.shape[1] for s in stack)
    return np.array([s[:min_rows, :min_cols] for s in stack])


def time_average(stack):
    """Compute the time-averaged displacement map, ignoring NaNs."""
    return np.nanmean(stack, axis=0)


def fit_affine_3d(x0, y0, z0, x1, y1, z1):
    """
    Fit a 4x4 affine transformation matrix mapping (x0, y0, z0) -> (x1, y1, z1).

    Returns the 4x4 matrix T such that:
        [x1, y1, z1, 1]^T = T @ [x0, y0, z0, 1]^T
    """
    n = len(x0)
    M = np.column_stack([x0, y0, z0, np.ones(n)])
    Ax, _, _, _ = np.linalg.lstsq(M, x1, rcond=None)
    Ay, _, _, _ = np.linalg.lstsq(M, y1, rcond=None)
    Az, _, _, _ = np.linalg.lstsq(M, z1, rcond=None)
    T = np.array([Ax, Ay, Az, [0, 0, 0, 1]])
    return T


def compute_residuals(T, x0, y0, z0, x1, y1, z1):
    """Compute residuals between predicted and actual deformed positions."""
    M = np.column_stack([x0, y0, z0, np.ones(len(x0))])
    x1_pred = M @ T[0]
    y1_pred = M @ T[1]
    z1_pred = M @ T[2]
    return x1 - x1_pred, y1 - y1_pred, z1 - z1_pred


def build_correspondences(undef_u, undef_v, undef_w, def_u, def_v, def_w):
    """
    Build 3D point correspondences from displacement fields.

    Each valid DIC grid point at position (col, row) has:
        Undeformed 3D position: (col + u_undef, row + v_undef, w_undef)
        Deformed 3D position:   (col + u_def,   row + v_def,   w_def)
    """
    valid = ~(np.isnan(undef_u) | np.isnan(undef_v) | np.isnan(undef_w) |
              np.isnan(def_u) | np.isnan(def_v) | np.isnan(def_w))

    rows, cols = np.where(valid)
    x = cols.astype(float)
    y = rows.astype(float)

    x0 = x + undef_u[valid]
    y0 = y + undef_v[valid]
    z0 = undef_w[valid]

    x1 = x + def_u[valid]
    y1 = y + def_v[valid]
    z1 = def_w[valid]

    return x0, y0, z0, x1, y1, z1, np.sum(valid)


def main():
    base = "processed_data/MatchID"

    # Load all frames
    print("Loading displacement fields...")
    undef_u_all = load_displacement_stack(f"{base}/undeformed_exp/u")
    undef_v_all = load_displacement_stack(f"{base}/undeformed_exp/v")
    undef_w_all = load_displacement_stack(f"{base}/undeformed_exp/w")
    def_u_all = load_displacement_stack(f"{base}/deformed_exp/u")
    def_v_all = load_displacement_stack(f"{base}/deformed_exp/v")
    def_w_all = load_displacement_stack(f"{base}/deformed_exp/w")

    n_frames = undef_u_all.shape[0]
    print(f"  {n_frames} frames per dataset")

    # --- Full fit using all frames ---
    print("\n=== Full fit (all frames) ===")
    undef_u = time_average(undef_u_all)
    undef_v = time_average(undef_v_all)
    undef_w = time_average(undef_w_all)
    def_u = time_average(def_u_all)
    def_v = time_average(def_v_all)
    def_w = time_average(def_w_all)

    x0, y0, z0, x1, y1, z1, n_pts = build_correspondences(
        undef_u, undef_v, undef_w, def_u, def_v, def_w)
    print(f"  Valid correspondence points: {n_pts}")

    T = fit_affine_3d(x0, y0, z0, x1, y1, z1)
    print(f"\n3D Affine Transformation Matrix (4x4):")
    print(np.array2string(T, precision=8, suppress_small=False))

    res_u, res_v, res_w = compute_residuals(T, x0, y0, z0, x1, y1, z1)
    rms = np.sqrt(np.mean(res_u**2 + res_v**2 + res_w**2))
    print(f"\nResiduals (fit):")
    print(f"  u: std={np.std(res_u):.6e} mm, max={np.max(np.abs(res_u)):.6e} mm")
    print(f"  v: std={np.std(res_v):.6e} mm, max={np.max(np.abs(res_v)):.6e} mm")
    print(f"  w: std={np.std(res_w):.6e} mm, max={np.max(np.abs(res_w)):.6e} mm")
    print(f"  total RMS: {rms:.6e} mm")

    # --- Split validation ---
    print("\n=== Split validation ===")
    for name, train_sl, test_sl in [
        ("Train 0-49, test 50-99", slice(0, 50), slice(50, 100)),
        ("Train 50-99, test 0-49", slice(50, 100), slice(0, 50)),
    ]:
        # Train
        x0_tr, y0_tr, z0_tr, x1_tr, y1_tr, z1_tr, _ = build_correspondences(
            time_average(undef_u_all[train_sl]),
            time_average(undef_v_all[train_sl]),
            time_average(undef_w_all[train_sl]),
            time_average(def_u_all[train_sl]),
            time_average(def_v_all[train_sl]),
            time_average(def_w_all[train_sl]),
        )
        T_split = fit_affine_3d(x0_tr, y0_tr, z0_tr, x1_tr, y1_tr, z1_tr)

        # Test
        x0_te, y0_te, z0_te, x1_te, y1_te, z1_te, n_te = build_correspondences(
            time_average(undef_u_all[test_sl]),
            time_average(undef_v_all[test_sl]),
            time_average(undef_w_all[test_sl]),
            time_average(def_u_all[test_sl]),
            time_average(def_v_all[test_sl]),
            time_average(def_w_all[test_sl]),
        )
        res_u, res_v, res_w = compute_residuals(T_split, x0_te, y0_te, z0_te,
                                                 x1_te, y1_te, z1_te)
        rms = np.sqrt(np.mean(res_u**2 + res_v**2 + res_w**2))
        print(f"\n{name} ({n_te} test points):")
        print(f"  u: std={np.std(res_u):.6e} mm")
        print(f"  v: std={np.std(res_v):.6e} mm")
        print(f"  w: std={np.std(res_w):.6e} mm")
        print(f"  total RMS: {rms:.6e} mm")

    # Save the matrix
    np.save("processed_data/MatchID/deformation_matrix_4x4.npy", T)
    print(f"\nMatrix saved to processed_data/MatchID/deformation_matrix_4x4.npy")


if __name__ == "__main__":
    main()
