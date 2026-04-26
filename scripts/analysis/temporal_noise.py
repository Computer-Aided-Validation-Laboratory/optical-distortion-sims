"""
Compute temporal displacement noise floor from DIC data.

For each spatial point, computes the standard deviation of U, V, W, u, v
across all time steps. Reports separate noise floor values for each component.

Usage:
    python temporal_noise.py undeformed
    python temporal_noise.py deformed
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.join(os.path.dirname(__file__), os.pardir)
DATASETS = {
    "undeformed": os.path.join(BASE_DIR, "processed_data", "experimental_undeformed"),
    "deformed": os.path.join(BASE_DIR, "processed_data", "experimental_deformed"),
}


def load_frames(data_dir):
    """Load all Image_XXXX_0.csv files, returning a list of DataFrames."""
    pattern = os.path.join(data_dir, "Image_*_0.csv")
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} frames in {data_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(f, skipinitialspace=True)
        df.columns = df.columns.str.strip().str.strip('"')
        frames.append(df)
    return frames


def compute_temporal_noise(frames):
    """
    Compute per-point temporal standard deviation of U, V, W, u, v.

    Points are matched across frames by their row index (pixel positions
    x, y are consistent across frames). Points with sigma == -1 in any
    frame are excluded.
    """
    n_frames = len(frames)
    n_points = len(frames[0])

    U_stack = np.zeros((n_frames, n_points))
    V_stack = np.zeros((n_frames, n_points))
    W_stack = np.zeros((n_frames, n_points))
    u_stack = np.zeros((n_frames, n_points))
    v_stack = np.zeros((n_frames, n_points))
    sigma_stack = np.zeros((n_frames, n_points))

    for i, df in enumerate(frames):
        U_stack[i] = df["U"].values
        V_stack[i] = df["V"].values
        W_stack[i] = df["W"].values
        u_stack[i] = df["u"].values
        v_stack[i] = df["v"].values
        sigma_stack[i] = df["sigma"].values

    valid = np.all(sigma_stack != -1, axis=0)
    print(f"Valid points (sigma != -1 in all frames): {valid.sum()} / {n_points}")

    x = frames[0]["x"].values
    y = frames[0]["y"].values

    std_U = np.full(n_points, np.nan)
    std_V = np.full(n_points, np.nan)
    std_W = np.full(n_points, np.nan)
    std_u = np.full(n_points, np.nan)
    std_v = np.full(n_points, np.nan)

    std_U[valid] = np.std(U_stack[:, valid], axis=0, ddof=1)
    std_V[valid] = np.std(V_stack[:, valid], axis=0, ddof=1)
    std_W[valid] = np.std(W_stack[:, valid], axis=0, ddof=1)
    std_u[valid] = np.std(u_stack[:, valid], axis=0, ddof=1)
    std_v[valid] = np.std(v_stack[:, valid], axis=0, ddof=1)

    stacks = [
        ("U", U_stack, "mm"), ("V", V_stack, "mm"), ("W", W_stack, "mm"),
        ("u", u_stack, "px"), ("v", v_stack, "px"),
    ]

    return x, y, std_U, std_V, std_W, std_u, std_v, valid, stacks


def print_stats(stacks, valid):
    """Print summary statistics for each displacement component."""
    for name, stack, unit in stacks:
        all_vals = stack[:, valid].ravel()
        noise_floor = np.std(all_vals, ddof=1)
        print(f"\n--- Temporal Noise Floor: {name} ---")
        print(f"  σ = {noise_floor:.6f} {unit}")
        if unit == "mm":
            print(f"  σ = {noise_floor*1000:.3f} µm")


def plot_noise_maps(x, y, std_U, std_V, std_W, std_u, std_v, valid, fig_dir, dataset_name):
    """Plot spatial noise maps and histograms for U, V, W, u, v."""
    os.makedirs(fig_dir, exist_ok=True)

    plot_configs = [
        ("U", std_U, "µm", 1000), ("V", std_V, "µm", 1000), ("W", std_W, "µm", 1000),
        ("u", std_u, "px", 1), ("v", std_v, "px", 1),
    ]

    for name, data, unit, scale in plot_configs:
        vals = data[valid]
        xv = x[valid]
        yv = y[valid]

        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(xv, yv, c=vals * scale, s=0.5, cmap="hot")
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        ax.set_title(f"Temporal Noise Floor ({dataset_name}): σ({name}) [{unit}]")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        plt.colorbar(sc, ax=ax, label=f"σ ({unit})")
        plt.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"noise_map_{name}_{dataset_name}.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(vals * scale, bins=100, edgecolor="black", linewidth=0.3)
        ax.set_xlabel(f"σ({name}) [{unit}]")
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of Temporal Noise ({dataset_name}): {name}")
        ax.axvline(vals.std() * scale, color="r", linestyle="--",
                    label=f"Std Dev = {vals.std()*scale:.4f} {unit}")
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"noise_hist_{name}_{dataset_name}.png"), dpi=150)
        plt.close(fig)

    print(f"\nPlots saved to {fig_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Compute temporal noise floor from DIC data.")
    parser.add_argument("dataset", choices=["undeformed", "deformed"],
                        help="Which dataset to analyse")
    args = parser.parse_args()

    data_dir = DATASETS[args.dataset]
    fig_dir = os.path.join(BASE_DIR, "figures")

    print(f"=== {args.dataset.upper()} ===")
    frames = load_frames(data_dir)
    x, y, std_U, std_V, std_W, std_u, std_v, valid, stacks = compute_temporal_noise(frames)
    print_stats(stacks, valid)
    plot_noise_maps(x, y, std_U, std_V, std_W, std_u, std_v, valid, fig_dir, args.dataset)


if __name__ == "__main__":
    main()
