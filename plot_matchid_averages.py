import numpy as np
import matplotlib.pyplot as plt
import os
import glob

base = "processed_data/MatchID"

datasets = {
    "Undeformed u (mm)": f"{base}/undeformed_exp/u",
    "Undeformed v (mm)": f"{base}/undeformed_exp/v",
    "Deformed u (mm)": f"{base}/deformed_exp/u",
    "Deformed v (mm)": f"{base}/deformed_exp/v",
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (label, folder) in enumerate(datasets.items()):
    csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    print(f"\n{label}: {len(csv_files)} files from {folder}")

    # Stack all 100 matrices
    stack = []
    for f in csv_files:
        data = np.genfromtxt(f, delimiter=",")
        # Some rows may have trailing comma -> extra NaN column, trim it
        stack.append(data)

    # Ensure consistent shape
    min_rows = min(s.shape[0] for s in stack)
    min_cols = min(s.shape[1] for s in stack)
    stack = np.array([s[:min_rows, :min_cols] for s in stack])  # (100, rows, cols)

    # Time-average (mean over the 100 images, ignoring NaN)
    mean_map = np.nanmean(stack, axis=0)
    std_map = np.nanstd(stack, axis=0)

    # Statistics (over all pixels, ignoring NaN)
    overall_mean = np.nanmean(mean_map)
    overall_std = np.nanmean(std_map)
    print(f"  Mean of time-averaged map: {overall_mean:.6e}")
    print(f"  Mean of temporal std map:  {overall_std:.6e}")

    # Plot
    ax = axes[idx]
    im = ax.imshow(mean_map, cmap="RdBu_r", aspect="equal", interpolation="nearest")
    ax.set_title(f"{label}\nmean={overall_mean:.4e}, std={overall_std:.4e}", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.subplots_adjust(hspace=0.35)
plt.savefig("figures/matchid_time_averaged_maps.png", dpi=200, bbox_inches="tight")
plt.show()
print("\nSaved to figures/matchid_time_averaged_maps.png")
