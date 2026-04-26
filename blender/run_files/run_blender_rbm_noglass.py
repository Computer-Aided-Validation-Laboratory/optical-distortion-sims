# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Diagnostic-only RBM render: no glass window.

Used to isolate whether the residual ~1.87 px epipolar error (when correlating
Blender RBM against calibration_1.caldat) comes from the glass model or from
elsewhere in the pipeline. Renders the speckled sample undergoing rigid body
motion with no glass window in the scene at all.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path
import bpy

# Pyvale imports
import pyvale.sensorsim as sens
import pyvale.dataset as dataset
import pyvale.blender as blender
import pyvale.mooseherder as mh

# %%
# The simulation results are loaded in here in the same way as the previous
# example. As mentioned this `data_path` can be replaced with your own MOOSE
# simulation output in exodus format (*.e).

data_path1 = dataset.render_mechanical_3d_path()
sim_data1 = mh.ExodusLoader(data_path1).load_all_sim_data()

# %%
# This is then scaled to mm, as all lengths in Blender are to be set in mm.

disp_comps = ("disp_x","disp_y", "disp_z")
sim_data1 = sens.scale_length_units(scale=1000.0,
                                     sim_data=sim_data1,
                                     disp_keys=disp_comps)

render_mesh1 = sens.create_render_mesh(sim_data1,
                                        ("disp_y","disp_x"),
                                        sim_spat_dim=3,
                                        field_disp_keys=disp_comps)

# %%
# Save path

base_dir = Path.cwd() / "blender/glass/rbm_noglass"

# %%
# Creating the scene
scene = blender.Scene()

sample = scene.add_cal_target(target_size=np.array([50, 35, 10]))
sample_location = np.array([0, 0, 0])
blender.Tools.move_blender_obj(part=sample, pos_world=sample_location)


calib_path = Path.cwd() / "blender/calibration.yaml"
pos_world_0 = (-17.5, 11.2, 393.5)
rot_world_0 = Rotation.from_euler("xyz", [0, -2.3, 0], degrees=True)
focal_length = 50
stereo_system = sens.camerastereo.CameraStereo.from_calibration(calib_path, pos_world_0, rot_world_0, focal_length, pixels_num=np.array([5328, 4608]))
cam0, cam1 = scene.add_stereo_system(stereo_system)

stereo_system.save_calibration(base_dir)

light_data = blender.LightData(type=blender.LightType.POINT,
                                     pos_world=(200, 0, 200),
                                     rot_world=Rotation.from_euler("xyz",
                                                                   [0, 0.8, 0]),
                                     energy=3.5)
light = scene.add_light(light_data)

# Apply the speckle pattern
material_data = blender.MaterialData()
speckle_path = Path.cwd() / "blender/speckle2.tiff"

mm_px_resolution = sens.CameraTools.calculate_mm_px_resolution(stereo_system.cam_data_0)
scene.add_speckle(part=sample,
                  speckle_path=speckle_path,
                  mat_data=material_data,
                  mm_px_resolution=mm_px_resolution,
                  cal=True)

# %%
# Render parameters
render_data = blender.RenderData(cam_data=(stereo_system.cam_data_0,
                                            stereo_system.cam_data_1),
                                base_dir=base_dir,
                                threads=240,
                                samples=256,
                                apply_distortion=True)

# Rigid body motion images (no glass deformation step — no glass in scene)
for i in range(0, 11):
    x_loc = i/100
    sample_location = np.array([x_loc, 0, 0])
    blender.Tools.move_blender_obj(part=sample, pos_world=sample_location)
    scene.render_single_image(render_data=render_data,
                          stage_image=False)
    file = base_dir / "images/blenderimage_0_0.tiff"
    newfilename =  "images/blenderimage_" + str(i) + "_0.tiff"
    file.rename(base_dir / newfilename)
    file1 = base_dir / "images/blenderimage_0_1.tiff"
    newfilename1 = "images/blenderimage_" + str(i) + "_1.tiff"
    file1.rename(base_dir / newfilename1)


# %%
# Save Blender project file
blender.Tools.save_blender_file(base_dir, over_write=True)
