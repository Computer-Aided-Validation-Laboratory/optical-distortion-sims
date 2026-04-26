"""Visualise MatchID epipolar distance maps: deformed vs undeformed comparison."""

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


def print_stats(label, frames, sorted_names):
    print(f"\n=== {label} ===")
    for name in sorted_names:
        d = frames[name]
        print(f"  {name}: mean={np.nanmean(d):.4f}, median={np.nanmedian(d):.4f}, "
              f"max={np.nanmax(d):.4f}, std={np.nanstd(d):.4f}")


# Load both datasets
def_frames, def_names = load_epi_dist("processed_data/MatchID/rbm/epi_dist")
undef_frames, undef_names = load_epi_dist("processed_data/MatchID/rbm_undef/epi_dist")

print_stats("Deformed (RBM)", def_frames, def_names)
print_stats("Undeformed (RBM)", undef_frames, undef_names)

# Use common color scale across both
vmin, vmax = 0, 0.5

# --- Figure 1: Side-by-side spatial maps (frame 1 from each) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Pick frame 1 from each
def_key = def_names[0]  # blenderimage_1
undef_key = undef_names[0]  # blenderimage_1

im0 = axes[0].imshow(undef_frames[undef_key], cmap="hot", vmin=vmin, vmax=vmax,
                      aspect="auto", origin="upper")
axes[0].set_title(f"Undeformed — {undef_key}", fontsize=12)
axes[0].set_xlabel("Column")
axes[0].set_ylabel("Row")

im1 = axes[1].imshow(def_frames[def_key], cmap="hot", vmin=vmin, vmax=vmax,
                      aspect="auto", origin="upper")
axes[1].set_title(f"Deformed — {def_key}", fontsize=12)
axes[1].set_xlabel("Column")
axes[1].set_ylabel("Row")

fig.colorbar(im1, ax=axes, label="Epipolar distance (px)", shrink=0.8)
fig.suptitle("Epipolar Distance Comparison: Undeformed vs Deformed", fontsize=14)
plt.tight_layout()
plt.savefig("figures/epi_dist_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved comparison to figures/epi_dist_comparison.png")

# --- Figure 2: All deformed frames ---
fig2, axes2 = plt.subplots(2, 5, figsize=(20, 8))
axes2 = axes2.flatten()
for i, name in enumerate(def_names):
    im = axes2[i].imshow(def_frames[name], cmap="hot", vmin=vmin, vmax=vmax,
                         aspect="auto", origin="upper")
    axes2[i].set_title(name, fontsize=10)
    axes2[i].set_xlabel("Column")
    axes2[i].set_ylabel("Row")
fig2.suptitle("Epipolar Distance — Deformed (RBM) Case", fontsize=14)
fig2.colorbar(im, ax=axes2, label="Epipolar distance (px)", shrink=0.8)
plt.tight_layout()
plt.savefig("figures/epi_dist_spatial_maps_rbm.png", dpi=150, bbox_inches="tight")

# --- Figure 3: All undeformed frames ---
fig3, axes3 = plt.subplots(2, 5, figsize=(20, 8))
axes3 = axes3.flatten()
for i, name in enumerate(undef_names):
    im = axes3[i].imshow(undef_frames[name], cmap="hot", vmin=vmin, vmax=vmax,
                         aspect="auto", origin="upper")
    axes3[i].set_title(name, fontsize=10)
    axes3[i].set_xlabel("Column")
    axes3[i].set_ylabel("Row")
fig3.suptitle("Epipolar Distance — Undeformed (RBM) Case", fontsize=14)
fig3.colorbar(im, ax=axes3, label="Epipolar distance (px)", shrink=0.8)
plt.tight_layout()
plt.savefig("figures/epi_dist_spatial_maps_rbm_undef.png", dpi=150, bbox_inches="tight")

# --- Figure 4: Mean per frame for both ---
fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

def_means = [np.nanmean(def_frames[n]) for n in def_names]
undef_means = [np.nanmean(undef_frames[n]) for n in undef_names]
def_medians = [np.nanmedian(def_frames[n]) for n in def_names]
undef_medians = [np.nanmedian(undef_frames[n]) for n in undef_names]

x_def = range(len(def_names))
x_undef = range(len(undef_names))

ax1.plot(x_def, def_means, "o-", label="Deformed — Mean", color="red")
ax1.plot(x_undef, undef_means, "o-", label="Undeformed — Mean", color="blue")
ax1.plot(x_def, def_medians, "s--", label="Deformed — Median", color="red", alpha=0.5)
ax1.plot(x_undef, undef_medians, "s--", label="Undeformed — Median", color="blue", alpha=0.5)
ax1.set_xlabel("Frame number")
ax1.set_ylabel("Epipolar distance (px)")
ax1.set_title("Mean/Median Epipolar Distance per Frame")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Histograms overlaid
all_def = np.concatenate([def_frames[n].flatten() for n in def_names])
all_def = all_def[~np.isnan(all_def)]
all_undef = np.concatenate([undef_frames[n].flatten() for n in undef_names])
all_undef = all_undef[~np.isnan(all_undef)]

ax2.hist(all_undef, bins=100, alpha=0.5, label=f"Undeformed (mean={np.mean(all_undef):.3f})", color="blue")
ax2.hist(all_def, bins=100, alpha=0.5, label=f"Deformed (mean={np.mean(all_def):.3f})", color="red")
ax2.set_xlabel("Epipolar distance (px)")
ax2.set_ylabel("Count")
ax2.set_title("Histogram of Epipolar Distance")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/epi_dist_comparison_stats.png", dpi=150, bbox_inches="tight")
print("Saved comparison stats to figures/epi_dist_comparison_stats.png")

plt.show()
