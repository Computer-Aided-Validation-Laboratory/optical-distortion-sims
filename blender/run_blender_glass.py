# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Deforming a sample with stereo DIC
===================================================

This example takes you through creating stereo DIC scene, applying deformation
to the sample, and rendering images at each deformation timestep.

Test case: mechanical analysis of a plate with a hole loaded in tension.
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
sim_data1 = mh.ExodusReader(data_path1).read_all_sim_data()

data_path2 = Path.cwd() / "moose/input/circular_glass_out.e"
sim_data2 = mh.ExodusReader(data_path2).read_all_sim_data()

# %%
# This is then scaled to mm, as all lengths in Blender are to be set in mm.
# The `SimData` object is then converted into a `RenderMeshData` object, as
# this skins the mesh ready to be imported into Blender.
# The `disp_comps` are the expected direction of displacement. Since this is a
# 3D deformation test case, displacement is expected in the x, y and z directions.

disp_comps = ("disp_x","disp_y", "disp_z")
sim_data1 = sens.scale_length_units(scale=1000.0,
                                     sim_data=sim_data1,
                                     disp_comps=disp_comps)
sim_data2 = sens.scale_length_units(scale=1000.0,
                                     sim_data=sim_data2,
                                     disp_comps=disp_comps)

render_mesh1 = sens.create_render_mesh(sim_data1,
                                        ("disp_y","disp_x"),
                                        sim_spat_dim=3,
                                        field_disp_keys=disp_comps)

render_mesh2 = sens.create_render_mesh(sim_data2,
                                        ("disp_y","disp_x"),
                                        sim_spat_dim=3,
                                        field_disp_keys=disp_comps)
# %%
# Firstly, a save path must be set.
# In order to do this a base path must be set. Then all the generated files will
# be saved to a subfolder within this specified base directory
# (e.g. blenderimages).
# If no base directory is specified, it will be set as your home directory.

base_dir = Path.cwd() / "blender/glass"

# %%
# Creating the scene
# ^^^^^^^^^^^^^^^^^^
# In order to create a DIC setup in Blender, first a scene must be created.
# A scene is initialised using the `BlenderScene` class. All the subsequent
# objects and actions necessary are then methods of this class.
scene = blender.Scene()


part = scene.add_part(render_mesh1, sim_spat_dim=3)
window = scene.add_part(render_mesh2, sim_spat_dim=3)
# Set the part location
window_location = np.array([0, 0, 200])
blender.Tools.move_blender_obj(part=window, pos_world=window_location)
# Set part rotation
part_rotation = Rotation.from_euler("xyz", [0, 0, 0], degrees=True)
blender.Tools.rotate_blender_obj(part=part, rot_world=part_rotation)

cam_data_0 = sens.CameraData(pixels_num=np.array([5328, 4608]),
                               pixels_size=np.array([0.00345, 0.00345]),
                               pos_world=np.array([0, 0, 400]),
                               rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
                               roi_cent_world=(0, 0, 0),
                               focal_length=15.0)
# Set this to "symmetric" to get a symmetric stereo system or set this to
# "faceon" to get a face-on stereo system
stereo_setup = "faceon"
if stereo_setup == "symmetric":
    stereo_system = sens.CameraTools.symmetric_stereo_cameras(
        cam_data_0=cam_data_0,
        stereo_angle=15.0)
elif stereo_setup == "faceon":
    stereo_system = sens.CameraTools.faceon_stereo_cameras(
        cam_data_0=cam_data_0,
        stereo_angle=10.0)
else:
    raise ValueError(f"Unknown stereo_setup: {stereo_setup}")

cam0, cam1 = scene.add_stereo_system(stereo_system)

stereo_system.save_calibration(base_dir)

light_data = blender.LightData(type=blender.LightType.POINT,
                                     pos_world=(200, 0, 200),
                                     rot_world=Rotation.from_euler("xyz",
                                                                   [0, 0.8, 0]),
                                     energy=2)
light = scene.add_light(light_data)
# light.location = (200, 0, 200)
# light.rotation_euler = (0, 0.785, 0) # NOTE: The default is an XYZ Euler angle

# Apply the speckle pattern
material_data = blender.MaterialData()
speckle_path = dataset.dic_pattern_5mpx_path()
# NOTE: If you wish to use a bigger camera, you will need to generate a
# bigger speckle pattern generator


mm_px_resolution = sens.CameraTools.calculate_mm_px_resolution(cam_data_0)
scene.add_speckle(part=part,
                  speckle_path=speckle_path,
                  mat_data=material_data,
                  mm_px_resolution=mm_px_resolution)

# Adding the glass material
bpy.data.materials.new("Material.001")
bpy.data.materials["Material.001"].use_nodes = True
mat_nodes = bpy.data.materials["Material.001"].node_tree.nodes
glass = mat_nodes.new(type="ShaderNodeBsdfGlass")
glass.inputs["IOR"].default_value = 1.3777
inp = bpy.data.materials["Material.001"].node_tree.nodes["Material Output"].inputs["Surface"]
outp = bpy.data.materials["Material.001"].node_tree.nodes["Glass BSDF"].outputs["BSDF"]
bpy.data.materials["Material.001"].node_tree.links.new(inp,outp)
bpy.data.objects["Part.001"].active_material = bpy.data.materials["Material.001"]


# %%
# Deforming the sample and rendering images
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Once all the objects have been added to the scene, the sample can be deformed,
# and images can be rendered.
# Firstly, all the rendering parameters must be set, including parameters such as
# the number of threads to use.
# Differently to a 2D DIC system, both cameras' parameters must be specified in
# the `RenderData` object.
render_data = blender.RenderData(cam_data=(stereo_system.cam_data_0,
                                            stereo_system.cam_data_1),
                                base_dir=base_dir,
                                threads=8,
                                samples=40)

# scene.render_single_image(render_data=render_data,
#                           stage_image=False)

scene.render_deformed_images(render_mesh=render_mesh2,
                             sim_spat_dim=3,
                             render_data=render_data,
                             part=window,
                             stage_image=False)



# %%
# The rendered image will be saved to this filepath: 

print("Save directory of the image:", (render_data.base_dir / "blenderimages"))

# %%
# There is also the option to save the scene as a Blender project file.
# This file can be opened with the Blender GUI to view the scene.

blender.Tools.save_blender_file(base_dir, override=True)