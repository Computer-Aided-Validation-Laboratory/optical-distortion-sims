# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Validation of Blender optics - Blender rendering
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

data_path = Path.cwd() / "moose/input/circular_glass_out.e"
sim_data = mh.ExodusLoader(data_path).load_all_sim_data()

# %%
# This is then scaled to mm, as all lengths in Blender are to be set in mm.
# The `SimData` object is then converted into a `RenderMeshData` object, as
# this skins the mesh ready to be imported into Blender.
# The `disp_comps` are the expected direction of displacement. Since this is a
# 3D deformation test case, displacement is expected in the x, y and z directions.

disp_comps = ("disp_x","disp_y", "disp_z")
sim_data = sens.scale_length_units(scale=1000.0,
                                     sim_data=sim_data,
                                     disp_keys=disp_comps)
render_mesh = sens.create_render_mesh(sim_data,
                                        ("disp_y","disp_x"),
                                        sim_spat_dim=3,
                                        field_disp_keys=disp_comps)
# %%
# Firstly, a save path must be set.
# In order to do this a base path must be set. Then all the generated files will
# be saved to a subfolder within this specified base directory
# (e.g. blenderimages).
# If no base directory is specified, it will be set as your home directory.

base_dir = Path.cwd() / "validation/blender/output"

# %%
# Creating the scene
# ^^^^^^^^^^^^^^^^^^
# In order to create a DIC setup in Blender, first a scene must be created.
# A scene is initialised using the `BlenderScene` class. All the subsequent
# objects and actions necessary are then methods of this class.
scene = blender.Scene()

bpy.context.scene.world.use_nodes = True
bg = bpy.context.scene.world.node_tree.nodes['Background']
bg.inputs[0].default_value = (0.5, 0.5, 0.5, 1.0)  # Grey color

# Work out what is a good size so that it fills the whole FOV
target = scene.add_cal_target(target_size=np.array([90, 80, 10]))
target_location = np.array([0, 0, 0]) 
blender.Tools.move_blender_obj(part=target, pos_world=target_location)

window = scene.add_part(render_mesh, sim_spat_dim=3)
# Set the part location
window_location = np.array([0, 0, 125]) 
blender.Tools.move_blender_obj(part=window, pos_world=window_location)


cam_data_0 = sens.CameraData(pixels_num=np.array([2464, 2056]),
                               pixels_size=np.array([0.00345, 0.00345]),
                               pos_world=np.array([0, 0, 400]),
                               rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
                               roi_cent_world=(0, 0, 0),
                               focal_length=35.0)
cam_0 = scene.add_camera(cam_data_0)

light_data = blender.LightData(type=blender.LightType.POINT,
                                     pos_world=(200, 0, 200),
                                     rot_world=Rotation.from_euler("xyz",
                                                                   [0, 0.8, 0]),
                                     energy=5)
light = scene.add_light(light_data)


# Apply the speckle pattern
material_data = blender.MaterialData()
# NOTE: If you wish to use a bigger camera, you will need to generate a
# bigger speckle pattern generator

rows, cols = 80, 90
square_size = 5
n_rows = rows // square_size
n_cols = cols // square_size
checkerboard_single = np.indices((n_rows, n_cols)).sum(axis=0) % 2
checkerboard = np.repeat(np.repeat(checkerboard_single, square_size, axis=0), square_size, axis=1)
checkerboard = (checkerboard * 255).astype(np.uint8)

mm_px_resolution = sens.CameraTools.calculate_mm_px_resolution(cam_data_0)
scene.add_speckle(part=target,
                  speckle_path=None,
                  speckle_pattern=checkerboard,
                  mat_data=material_data,
                  mm_px_resolution=mm_px_resolution,
                  cal=True)

# Adding the glass material
bpy.data.materials.new("Material.001")
bpy.data.materials["Material.001"].use_nodes = True
mat_nodes = bpy.data.materials["Material.001"].node_tree.nodes
glass = mat_nodes.new(type="ShaderNodeBsdfGlass")
glass.inputs["IOR"].default_value = 1.3777
inp = bpy.data.materials["Material.001"].node_tree.nodes["Material Output"].inputs["Surface"]
outp = bpy.data.materials["Material.001"].node_tree.nodes["Glass BSDF"].outputs["BSDF"]
bpy.data.materials["Material.001"].node_tree.links.new(inp,outp)
bpy.data.objects["Part"].active_material = bpy.data.materials["Material.001"]


print("Film transparent:", bpy.context.scene.render.film_transparent)
print("World nodes enabled:", bpy.context.scene.world.use_nodes)
print("Background color:", bg.inputs[0].default_value[:])
print("Background strength:", bg.inputs[1].default_value)


# %%
# Deforming the sample and rendering images
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Once all the objects have been added to the scene, the sample can be deformed,
# and images can be rendered.
# Firstly, all the rendering parameters must be set, including parameters such as
# the number of threads to use.
# Differently to a 2D DIC system, both cameras' parameters must be specified in
# the `RenderData` object.
render_data = blender.RenderData(cam_data=cam_data_0,
                                 base_dir=base_dir,
                                threads=16,
                                samples=1024)

# scene.render_single_image(render_data=render_data,
#                           stage_image=False)

scene.render_deformed_images(render_mesh=render_mesh,
                             sim_spat_dim=3,
                             render_data=render_data,
                             part=window,
                             stage_image=False)

# %%
# There is also the option to save the scene as a Blender project file.
# This file can be opened with the Blender GUI to view the scene.

blender.Tools.save_blender_file(base_dir, over_write=True)