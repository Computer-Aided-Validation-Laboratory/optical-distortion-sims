"""Plot corrected calibration epipolar distance and difference vs deformed_exp."""

import glob
import os
import numpy as np
import matplotlib.pyplot as plt


def load_epi_dist(data_dir):
    """Load all epipolar distance CSVs from a directory."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    frames = {}
    for f in files:
        name = os.path.basename(f).replace(".tiff_epi_dist.csv", "")
        data = np.genfromtxt(f, delimiter=",")
        data = data[:, :-1]  # drop trailing NaN column from trailing comma
        frames[name] = data
    sorted_names = sorted(frames.keys(), key=lambda x: int(x.split("_")[1]))
    return frames, sorted_names


# Load both datasets
corr_frames, corr_names = load_epi_dist("processed_data/MatchID/corrected_calib/epi_dist")
def_frames, def_names = load_epi_dist("processed_data/MatchID/deformed_exp/epi_dist")

# Compute time-averaged maps
corr_stack = np.stack([corr_frames[n] for n in corr_names], axis=0)
def_stack = np.stack([def_frames[n] for n in def_names], axis=0)
corr_mean = np.nanmean(corr_stack, axis=0)
def_mean = np.nanmean(def_stack, axis=0)

# Print stats
print(f"Corrected calib: mean={np.nanmean(corr_mean):.4f}, "
      f"median={np.nanmedian(corr_mean):.4f}, max={np.nanmax(corr_mean):.4f}")
print(f"Deformed exp:    mean={np.nanmean(def_mean):.4f}, "
      f"median={np.nanmedian(def_mean):.4f}, max={np.nanmax(def_mean):.4f}")

# Crop to matching shape (corrected_calib has 114 rows, deformed_exp has 115)
min_rows = min(corr_mean.shape[0], def_mean.shape[0])
min_cols = min(corr_mean.shape[1], def_mean.shape[1])
diff = def_mean[:min_rows, :min_cols] - corr_mean[:min_rows, :min_cols]

# --- Figure: Corrected calib (left) and difference (right) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

im1 = ax1.imshow(corr_mean, cmap="hot", vmin=0,
                 aspect="auto", origin="upper")
ax1.set_title("Corrected Calibration", fontsize=13)
ax1.set_xlabel("Column")
ax1.set_ylabel("Row")
fig.colorbar(im1, ax=ax1, label="Epipolar distance (px)", shrink=0.8)

im2 = ax2.imshow(diff, cmap="hot", vmin=0,
                 aspect="auto", origin="upper")
ax2.set_title("Deformed — Corrected", fontsize=13)
ax2.set_xlabel("Column")
ax2.set_ylabel("Row")
fig.colorbar(im2, ax=ax2, label="Difference (px)", shrink=0.8)

fig.suptitle("Time-Averaged Epipolar Distance", fontsize=14)
plt.tight_layout()
fig.savefig("figures/epi_dist_corrected_calib.png", dpi=150, bbox_inches="tight")
print("Saved figures/epi_dist_corrected_calib.png")

print(f"\nDifference stats: mean={np.nanmean(diff):.4f}, "
      f"median={np.nanmedian(diff):.4f}, min={np.nanmin(diff):.4f}, max={np.nanmax(diff):.4f}")

plt.show()
