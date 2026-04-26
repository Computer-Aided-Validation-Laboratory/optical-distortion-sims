"""Generate figures comparing experimental and Blender MatchID results."""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os


def load_epi_dist_mean(data_dir):
    """Load all epipolar distance CSVs and return mean map + overall stats."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    frames = []
    for f in files:
        data = np.genfromtxt(f, delimiter=",")
        data = data[:, :-1]  # trailing comma
        frames.append(data)
    mean_map = np.nanmean(np.array(frames), axis=0)
    per_frame_means = [np.nanmean(f) for f in frames]
    per_frame_medians = [np.nanmedian(f) for f in frames]
    return mean_map, np.mean(per_frame_means), np.mean(per_frame_medians), len(files)


def load_disp_mean(folder):
    """Load all displacement CSVs and return time-averaged map."""
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    stack = []
    for f in files:
        data = np.genfromtxt(f, delimiter=",")
        stack.append(data)
    min_rows = min(s.shape[0] for s in stack)
    min_cols = min(s.shape[1] for s in stack)
    stack = np.array([s[:min_rows, :min_cols] for s in stack])
    return np.nanmean(stack, axis=0)


base = "processed_data/MatchID"

# === FIGURE 1: Epipolar distance spatial maps (2x2, independent scales) ===
print("Loading epipolar distance data...")
datasets_epi = {
    "Experimental\nUndeformed": f"{base}/undeformed_exp/epi_dist",
    "Experimental\nDeformed": f"{base}/deformed_exp/epi_dist",
    "Blender\nUndeformed": f"{base}/new_rendering/rbm_undeformed_noisy/epi_dist",
    "Blender\nDeformed": f"{base}/new_rendering/rbm_deformed_noisy/epi_dist",
}

epi_maps = {}
epi_stats = {}
for label, path in datasets_epi.items():
    mean_map, mean_val, median_val, n = load_epi_dist_mean(path)
    epi_maps[label] = mean_map
    epi_stats[label] = (mean_val, median_val, n)
    print(f"  {label.replace(chr(10), ' ')}: {n} frames, mean={mean_val:.4f}, median={median_val:.4f}")

labels_order = [
    "Experimental\nUndeformed", "Experimental\nDeformed",
    "Blender\nUndeformed", "Blender\nDeformed",
]

# Use independent colour scales: one for experimental row, one for Blender row
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))

vmax_vals = [0.6, 0.6, 0.05, 0.05]

for idx, label in enumerate(labels_order):
    row, col = divmod(idx, 2)
    ax = axes1[row, col]
    mean_val, median_val, n = epi_stats[label]
    im = ax.imshow(epi_maps[label], cmap="hot", vmin=0, vmax=vmax_vals[idx],
                   aspect="auto", origin="upper", interpolation="nearest")
    ax.set_title(f"{label}", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"mean = {mean_val:.3f} px", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="px")

plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.savefig("figures/results_epi_dist_comparison.png", dpi=200, bbox_inches="tight")
print("Saved figures/results_epi_dist_comparison.png")

# Print table
print("\nEpipolar distance summary table:")
print(f"{'Dataset':<30} {'Mean (px)':<12} {'Median (px)':<12}")
for label in labels_order:
    mean_val, median_val, n = epi_stats[label]
    print(f"{label.replace(chr(10), ' '):<30} {mean_val:<12.4f} {median_val:<12.4f}")

# === FIGURE 2: Time-averaged displacement maps (4x3, per-row colour scales) ===
print("\nLoading displacement data...")
datasets_disp = {
    "Experimental\nUndeformed": {
        "u": f"{base}/undeformed_exp/u",
        "v": f"{base}/undeformed_exp/v",
        "w": f"{base}/undeformed_exp/w",
    },
    "Experimental\nDeformed": {
        "u": f"{base}/deformed_exp/u",
        "v": f"{base}/deformed_exp/v",
        "w": f"{base}/deformed_exp/w",
    },
    "Blender\nUndeformed": {
        "u": f"{base}/new_rendering/rbm_undeformed_noisy/u",
        "v": f"{base}/new_rendering/rbm_undeformed_noisy/v",
        "w": f"{base}/new_rendering/rbm_undeformed_noisy/w",
    },
    "Blender\nDeformed": {
        "u": f"{base}/new_rendering/rbm_deformed_noisy/u",
        "v": f"{base}/new_rendering/rbm_deformed_noisy/v",
        "w": f"{base}/new_rendering/rbm_deformed_noisy/w",
    },
}

row_labels = list(datasets_disp.keys())
col_labels = ["u", "v", "w"]

disp_maps = {}
for row_label, components in datasets_disp.items():
    disp_maps[row_label] = {}
    for comp, path in components.items():
        mean_map = load_disp_mean(path)
        # Subtract the spatial mean to show deviation from mean (removes RBM offset)
        mean_map_centered = mean_map - np.nanmean(mean_map)
        disp_maps[row_label][comp] = mean_map_centered
        print(f"  {row_label.replace(chr(10), ' ')} {comp}: "
              f"raw mean={np.nanmean(mean_map):.4e}, std={np.nanstd(mean_map):.4e}")

fig2, axes2 = plt.subplots(4, 3, figsize=(16, 12))

for i, row_label in enumerate(row_labels):
    for j, comp in enumerate(col_labels):
        ax = axes2[i, j]
        data = disp_maps[row_label][comp]
        # Per-subplot symmetric colour scale
        vmax_abs = np.nanpercentile(np.abs(data), 99)
        im = ax.imshow(data, cmap="RdBu_r", aspect="equal", origin="upper",
                       interpolation="nearest",
                       vmin=-vmax_abs, vmax=vmax_abs)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_title(f"{comp} (mm)", fontsize=13, fontweight="bold")
        if j == 0:
            ax.set_ylabel(row_label, fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.subplots_adjust(hspace=0.15, wspace=0.35)
plt.savefig("figures/results_disp_comparison.png", dpi=200, bbox_inches="tight")
print("\nSaved figures/results_disp_comparison.png")

plt.show()
