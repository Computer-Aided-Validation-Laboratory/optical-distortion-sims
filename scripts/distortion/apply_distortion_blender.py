"""
Apply Brown-Conrady lens distortion using Blender's Movie Distortion compositor node.

Reads calibration parameters from a YAML file and applies forward distortion
to all images in the specified directories. Images ending in _0 use Cam0
parameters, images ending in _1 use Cam1 parameters.

The distortion is applied using Blender's compositor: for each image, a
MovieClip is created with the calibration's Brown-Conrady coefficients,
and a CompositorNodeMovieDistortion node applies the distortion.

Usage:
    python apply_distortion_blender.py --calib calibration/calibration.yaml \
        --dirs blender/glass/rbm/images --pixel-size 0.00256

    # Or via Blender's bundled Python:
    blender --background --python apply_distortion_blender.py -- \
        --calib calibration/calibration.yaml --dirs blender/glass/rbm/images
"""

import sys
import numpy as np
from pathlib import Path
import yaml
import argparse

import bpy


def load_calibration(calib_path: Path) -> dict:
    with open(calib_path, "r") as f:
        return yaml.safe_load(f)


def get_camera_id(filepath: Path) -> int:
    stem = filepath.stem
    cam_id = int(stem.rsplit("_", 1)[-1])
    return cam_id


def get_camera_params(params: dict,
                      cam_id: int,
                      pixel_size: float,
                      image_size: tuple[int, int]) -> dict:
    prefix = f"Cam{cam_id}"
    fx = params[f"{prefix}_Fx [pixels]"]
    cx = params[f"{prefix}_Cx [pixels]"]
    cy = params[f"{prefix}_Cy [pixels]"]

    w, h = image_size
    focal_length_mm = fx * pixel_size
    sensor_width_mm = w * pixel_size

    principal_x = (cx - w / 2) / w
    principal_y = (cy - h / 2) / h

    return {
        "focal_length_mm": focal_length_mm,
        "sensor_width_mm": sensor_width_mm,
        "principal_point": (principal_x, principal_y),
        "k1": params[f"{prefix}_Kappa 1"],
        "k2": params[f"{prefix}_Kappa 2"],
        "k3": params[f"{prefix}_Kappa 3"],
        "p1": params[f"{prefix}_P1"],
        "p2": params[f"{prefix}_P2"],
    }


def setup_compositor(image: bpy.types.Image,
                     clip: bpy.types.MovieClip,
                     cam_params: dict) -> None:
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    # Configure the clip's tracking camera with Brown-Conrady parameters
    tc = clip.tracking.camera
    tc.distortion_model = 'BROWN'
    tc.focal_length = cam_params["focal_length_mm"]
    tc.sensor_width = cam_params["sensor_width_mm"]
    tc.brown_k1 = cam_params["k1"]
    tc.brown_k2 = cam_params["k2"]
    tc.brown_k3 = cam_params["k3"]
    tc.brown_p1 = cam_params["p1"]
    tc.brown_p2 = cam_params["p2"]
    tc.principal_point = cam_params["principal_point"]

    # Image input node
    image_node = tree.nodes.new(type='CompositorNodeImage')
    image_node.image = image

    # Movie Distortion node
    distort_node = tree.nodes.new(type='CompositorNodeMovieDistortion')
    distort_node.clip = clip
    distort_node.distortion_type = 'DISTORT'

    # Composite output
    comp_node = tree.nodes.new(type='CompositorNodeComposite')

    # Wire: Image -> Distortion -> Composite
    tree.links.new(image_node.outputs['Image'], distort_node.inputs['Image'])
    tree.links.new(distort_node.outputs['Image'], comp_node.inputs['Image'])


def process_directory(image_dir: Path,
                      output_dir: Path,
                      params: dict,
                      pixel_size: float,
                      image_size: tuple[int, int]) -> None:
    cam_params_cache = {}

    image_files = sorted(image_dir.glob("*.tiff"))
    if not image_files:
        print(f"No .tiff images found in {image_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    scene.render.image_settings.file_format = "TIFF"
    scene.render.image_settings.color_mode = "BW"
    scene.render.image_settings.color_depth = "16"
    scene.render.use_compositing = True
    scene.render.use_sequencer = False

    for img_path in image_files:
        cam_id = get_camera_id(img_path)

        if cam_id not in cam_params_cache:
            cam_params_cache[cam_id] = get_camera_params(params, cam_id,
                                                         pixel_size,
                                                         image_size)

        cam_params = cam_params_cache[cam_id]

        # Load the image and create a movie clip for distortion metadata
        image = bpy.data.images.load(str(img_path))
        clip = bpy.data.movieclips.load(str(img_path))

        # Set render resolution to match the input image
        w, h = image.size
        scene.render.resolution_x = w
        scene.render.resolution_y = h
        scene.render.resolution_percentage = 100

        output_path = output_dir / img_path.name
        scene.render.filepath = str(output_path)

        setup_compositor(image, clip, cam_params)

        bpy.ops.render.render(write_still=True)

        # Cleanup to prevent memory buildup
        bpy.data.images.remove(image)
        bpy.data.movieclips.remove(clip)

        print(f"  {img_path.name} -> {output_path}")


def main():
    # When run via `blender --background --python script.py -- args`,
    # Blender's own args come before `--` and ours come after.
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Apply lens distortion using Blender's Movie Distortion node."
    )
    parser.add_argument(
        "--calib",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs/calibration.yaml",
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
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="Physical pixel size in mm. If not provided, derived from "
             "focal-length / Fx.",
    )
    parser.add_argument(
        "--focal-length",
        type=float,
        default=50.0,
        help="Camera focal length in mm (default: 50.0). Used to derive "
             "pixel size when --pixel-size is not given.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[5328, 4608],
        metavar=("WIDTH", "HEIGHT"),
        help="Image dimensions in pixels (default: 5328 4608)",
    )
    args = parser.parse_args(argv)

    params = load_calibration(args.calib)
    print(f"Loaded calibration from {args.calib}")

    if args.pixel_size is not None:
        pixel_size = args.pixel_size
    else:
        fx = params["Cam0_Fx [pixels]"]
        pixel_size = args.focal_length / fx

    image_size = tuple(args.image_size)

    print(f"Pixel size: {pixel_size:.6f} mm")
    print(f"Image size: {image_size[0]} x {image_size[1]}")

    for image_dir in args.dirs:
        output_dir = image_dir.parent / (image_dir.name + args.suffix)
        print(f"\nProcessing {image_dir} -> {output_dir}")
        process_directory(image_dir, output_dir, params, pixel_size, image_size)

    print("\nDone.")


if __name__ == "__main__":
    main()
