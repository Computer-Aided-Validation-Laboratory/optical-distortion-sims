"""
Apply Brown-Conrady lens distortion to undistorted Blender-rendered images.

Reads calibration parameters from a YAML file and applies forward distortion
to all images in the specified directories. Images ending in _0 use Cam0
parameters, images ending in _1 use Cam1 parameters.

The distortion is applied using OpenCV: for each pixel in the distorted output,
cv2.undistortPoints finds the corresponding location in the undistorted input,
and cv2.remap resamples the image.
"""

import cv2
import numpy as np
from pathlib import Path
import yaml
import argparse


def load_calibration(calib_path: Path) -> dict:
    with open(calib_path, "r") as f:
        return yaml.safe_load(f)


def get_camera_params(params: dict, cam_id: int) -> tuple[np.ndarray, np.ndarray]:
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

    return camera_matrix, dist_coeffs


def build_distortion_maps(camera_matrix: np.ndarray,
                          dist_coeffs: np.ndarray,
                          image_shape: tuple[int, int]
                          ) -> tuple[np.ndarray, np.ndarray]:
    h, w = image_shape[:2]

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

    return map_x, map_y


def apply_distortion(image: np.ndarray,
                     map_x: np.ndarray,
                     map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def get_camera_id(filepath: Path) -> int:
    stem = filepath.stem
    cam_id = int(stem.rsplit("_", 1)[-1])
    return cam_id


def process_directory(image_dir: Path,
                      output_dir: Path,
                      params: dict) -> None:
    cam_maps = {}

    image_files = sorted(image_dir.glob("*.tiff"))
    if not image_files:
        print(f"No .tiff images found in {image_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_files:
        cam_id = get_camera_id(img_path)
        image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Could not read {img_path}, skipping")
            continue

        if cam_id not in cam_maps:
            camera_matrix, dist_coeffs = get_camera_params(params, cam_id)
            map_x, map_y = build_distortion_maps(
                camera_matrix, dist_coeffs, image.shape
            )
            cam_maps[cam_id] = (map_x, map_y)

        map_x, map_y = cam_maps[cam_id]
        distorted = apply_distortion(image, map_x, map_y)

        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), distorted)
        print(f"  {img_path.name} -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply lens distortion to Blender-rendered images."
    )
    parser.add_argument(
        "--calib",
        type=Path,
        default=Path("calibration/calibration.yaml"),
        help="Path to calibration YAML file",
    )
    parser.add_argument(
        "--dirs",
        type=Path,
        nargs="+",
        default=[
            Path("blender/glass/rbm/images"),
            Path("blender/glass/rbm_undeformed/images"),
        ],
        help="Directories containing images to distort",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_distorted",
        help="Suffix to add to output directory name (default: _distorted)",
    )
    args = parser.parse_args()

    params = load_calibration(args.calib)
    print(f"Loaded calibration from {args.calib}")

    for image_dir in args.dirs:
        output_dir = image_dir.parent / (image_dir.name + args.suffix)
        print(f"\nProcessing {image_dir} -> {output_dir}")
        process_directory(image_dir, output_dir, params)

    print("\nDone.")


if __name__ == "__main__":
    main()
