"""
Compare OpenCV vs Blender Movie Distortion output on a single test image.

Applies Brown-Conrady distortion using both methods with the same calibration
parameters, then computes pixel-wise difference statistics and saves a
visual comparison figure.
"""

import cv2
import numpy as np
from pathlib import Path
import yaml
import matplotlib.pyplot as plt

import bpy


# ── Shared helpers ──────────────────────────────────────────────────────────

def load_calibration(calib_path: Path) -> dict:
    with open(calib_path, "r") as f:
        return yaml.safe_load(f)


# ── OpenCV distortion ──────────────────────────────────────────────────────

def opencv_distort(image: np.ndarray,
                   params: dict,
                   cam_id: int) -> np.ndarray:
    prefix = f"Cam{cam_id}"
    fx = params[f"{prefix}_Fx [pixels]"]
    fy = params[f"{prefix}_Fy [pixels]"]
    cx = params[f"{prefix}_Cx [pixels]"]
    cy = params[f"{prefix}_Cy [pixels]"]
    fs = params[f"{prefix}_Fs [pixels]"]

    camera_matrix = np.array([[fx, fs, cx],
                              [0,  fy, cy],
                              [0,  0,  1]], dtype=np.float64)

    k1 = params[f"{prefix}_Kappa 1"]
    k2 = params[f"{prefix}_Kappa 2"]
    k3 = params[f"{prefix}_Kappa 3"]
    p1 = params[f"{prefix}_P1"]
    p2 = params[f"{prefix}_P2"]
    dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

    h, w = image.shape[:2]
    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
    points = np.stack([u_coords.ravel(), v_coords.ravel()],
                      axis=-1).astype(np.float32)

    undistorted_pts = cv2.undistortPoints(
        points.reshape(-1, 1, 2),
        camera_matrix,
        dist_coeffs,
        P=camera_matrix,
    )

    map_x = undistorted_pts[:, 0, 0].reshape(h, w).astype(np.float32)
    map_y = undistorted_pts[:, 0, 1].reshape(h, w).astype(np.float32)

    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ── Blender distortion ─────────────────────────────────────────────────────

def blender_distort(image_path: Path,
                    output_path: Path,
                    params: dict,
                    cam_id: int,
                    focal_length: float = 50.0) -> np.ndarray:
    prefix = f"Cam{cam_id}"
    fx = params[f"{prefix}_Fx [pixels]"]
    cx = params[f"{prefix}_Cx [pixels]"]
    cy = params[f"{prefix}_Cy [pixels]"]

    pixel_size = focal_length / fx

    # Load the image to get its dimensions
    img = bpy.data.images.load(str(image_path))
    w, h = img.size
    bpy.data.images.remove(img)

    sensor_width_mm = w * pixel_size

    # Create MovieClip from the image file
    clip = bpy.data.movieclips.load(str(image_path))

    # Configure tracking camera with Brown-Conrady parameters
    tc = clip.tracking.camera
    tc.distortion_model = 'BROWN'
    # sensor_width MUST be set before focal_length
    tc.sensor_width = sensor_width_mm
    tc.focal_length = focal_length
    tc.brown_k1 = params[f"{prefix}_Kappa 1"]
    tc.brown_k2 = params[f"{prefix}_Kappa 2"]
    tc.brown_k3 = params[f"{prefix}_Kappa 3"]
    tc.brown_p1 = params[f"{prefix}_P1"]
    tc.brown_p2 = params[f"{prefix}_P2"]
    tc.principal_point = ((cx - w / 2) / w, (cy - h / 2) / h)

    # Set up compositor
    scene = bpy.context.scene
    scene.use_nodes = True
    scene.render.use_compositing = True
    scene.render.use_sequencer = False

    # Disable colour management so pixel values pass through unchanged
    scene.view_settings.view_transform = 'Raw'

    tree = scene.node_tree
    tree.nodes.clear()

    # Load image as compositor input with raw colour space
    image_node = tree.nodes.new(type='CompositorNodeImage')
    loaded_img = bpy.data.images.load(str(image_path))
    loaded_img.colorspace_settings.name = 'Non-Color'
    image_node.image = loaded_img

    distort_node = tree.nodes.new(type='CompositorNodeMovieDistortion')
    distort_node.clip = clip
    distort_node.distortion_type = 'DISTORT'

    comp_node = tree.nodes.new(type='CompositorNodeComposite')

    tree.links.new(image_node.outputs['Image'],
                   distort_node.inputs['Image'])
    tree.links.new(distort_node.outputs['Image'],
                   comp_node.inputs['Image'])

    # Render settings
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "TIFF"
    scene.render.image_settings.color_mode = "BW"
    scene.render.image_settings.color_depth = "8"
    scene.render.filepath = str(output_path)

    bpy.ops.render.render(write_still=True)

    # Cleanup
    for img in list(bpy.data.images):
        bpy.data.images.remove(img)
    for mc in list(bpy.data.movieclips):
        bpy.data.movieclips.remove(mc)
    scene.use_nodes = False

    return cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)


# ── Main comparison ─────────────────────────────────────────────────────────

def main():
    repo_root = Path(__file__).resolve().parents[2]
    calib_path = repo_root / "configs/calibration.yaml"
    params = load_calibration(calib_path)

    # Use first Cam0 image (lives in the sibling optical-distortion-images repo)
    test_image_path = repo_root.parent / "optical-distortion-images/blender/glass/rbm/images/blenderimage_1_0.tiff"
    cam_id = 0

    print(f"Test image: {test_image_path}")
    print(f"Camera: Cam{cam_id}")

    # OpenCV distortion
    print("\nRunning OpenCV distortion...")
    image = cv2.imread(str(test_image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"ERROR: Could not read {test_image_path}")
        return
    print(f"  Image shape: {image.shape}, dtype: {image.dtype}")
    result_opencv = opencv_distort(image, params, cam_id)

    figures_dir = repo_root / "figures"
    figures_dir.mkdir(exist_ok=True)
    opencv_out = figures_dir / "comparison_opencv.tiff"
    cv2.imwrite(str(opencv_out), result_opencv)
    print(f"  Saved to {opencv_out}")

    # Blender distortion
    print("\nRunning Blender distortion...")
    blender_out = figures_dir / "comparison_blender.tiff"
    result_blender = blender_distort(test_image_path, blender_out,
                                     params, cam_id)
    print(f"  Saved to {blender_out}")

    # Compare - normalise both to [0, 1] to handle bit depth differences
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    print(f"  OpenCV output:  dtype={result_opencv.dtype}, shape={result_opencv.shape}")
    print(f"  Blender output: dtype={result_blender.dtype}, shape={result_blender.shape}")

    ocv_max = 255.0 if result_opencv.dtype == np.uint8 else 65535.0
    bld_max = 255.0 if result_blender.dtype == np.uint8 else 65535.0

    ocv = result_opencv.astype(np.float64) / ocv_max
    bld = result_blender.astype(np.float64) / bld_max

    if ocv.shape != bld.shape:
        print(f"  Shape mismatch: OpenCV {ocv.shape} vs Blender {bld.shape}")
        return

    diff = np.abs(ocv - bld)

    print(f"\n  Normalised to [0, 1]:")
    print(f"  Max absolute diff:  {diff.max():.6f} ({diff.max()*100:.4f}%)")
    print(f"  Mean absolute diff: {diff.mean():.6f} ({diff.mean()*100:.4f}%)")
    print(f"  Std of diff:        {diff.std():.6f}")
    print(f"  Median diff:        {np.median(diff):.6f}")
    print(f"  99th percentile:    {np.percentile(diff, 99):.6f}")

    # Pixels with near-zero difference (within quantisation noise)
    near_zero_pct = (diff < 1/256).sum() / diff.size * 100
    print(f"  Near-identical pixels (<1/256): {near_zero_pct:.1f}%")

    # Use normalised values for plotting
    result_opencv_plot = ocv
    result_blender_plot = bld

    # Save difference visualisation
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    axes[0, 0].imshow(result_opencv_plot, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title("OpenCV distortion")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(result_blender_plot, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title("Blender Movie Distortion")
    axes[0, 1].axis('off')

    im = axes[1, 0].imshow(diff, cmap='hot')
    axes[1, 0].set_title(f"Absolute difference (max={diff.max():.1f})")
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    # Histogram of non-zero differences
    nonzero_diff = diff[diff > 0]
    if len(nonzero_diff) > 0:
        axes[1, 1].hist(nonzero_diff.ravel(), bins=100, color='steelblue',
                        edgecolor='none')
        axes[1, 1].set_xlabel("Absolute pixel difference")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_title("Distribution of non-zero differences")
    else:
        axes[1, 1].text(0.5, 0.5, "All pixels identical",
                        ha='center', va='center', fontsize=14)

    fig_path = figures_dir / "distortion_comparison.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    print(f"\nComparison figure saved to {fig_path}")


if __name__ == "__main__":
    main()
