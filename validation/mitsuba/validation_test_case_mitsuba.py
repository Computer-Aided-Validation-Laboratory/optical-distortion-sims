"""
Validation of Blender optics - Mitsuba rendering
Recreates the pyvale Blender test case in Mitsuba 3
"""

import numpy as np
import mitsuba
from pathlib import Path
import matplotlib.pyplot as plt
import pyvale
import pyvale.mooseherder as mh
import pyvista as pv
import os

def cleanup_ply_header(filename):
    """Removes 'obj_info' lines from the PLY header that crash Mitsuba."""
    with open(filename, 'rb') as f:
        lines = f.readlines()

    with open(filename, 'wb') as f:
        header_ended = False
        for line in lines:
            if not header_ended:
                if b'obj_info' in line:
                    continue 
                if b'end_header' in line:
                    header_ended = True
            
            f.write(line)

def export_moose_to_mitsuba(data_path, output_filename):
    sim_data2 = mh.ExodusLoader(data_path).load_all_sim_data()
    disp_comps = ("disp_x","disp_y", "disp_z")
    sim_data2 = pyvale.sensorsim.scale_length_units(scale=1000.0,
                                        sim_data=sim_data2,
                                        disp_keys=disp_comps)
    render_mesh2 = pyvale.sensorsim.create_render_mesh(sim_data2,
                                            ("disp_y","disp_x"),
                                            sim_spat_dim=3,
                                            field_disp_keys=disp_comps)
    
    timestep = render_mesh2.fields_render.shape[1] - 1
    # timestep = 0
    deformed_nodes = pyvale.sensorsim.get_deformed_nodes(timestep,
                                                         render_mesh2)
    
    nodes_centred = pyvale.sensorsim.centre_mesh_nodes(deformed_nodes,
                                              spat_dim=3)
    faces = render_mesh2.connectivity
    padding = np.full((faces.shape[0], 1), 4)
    faces_with_padding = np.hstack((padding, faces))
    faces_pv = faces_with_padding.ravel().astype(np.int_)

    mesh = pv.PolyData(nodes_centred, faces_pv)

    surf_mesh = mesh.extract_surface()
    tri_mesh = surf_mesh.triangulate()
    tri_mesh.field_data.clear() 
    tri_mesh.save(output_filename, binary=True)
    cleanup_ply_header(output_filename)
    
    print(f"Exported triangle mesh: {tri_mesh.n_cells} triangles.")
    return output_filename


def main():
    os.environ["DRJIT_NUM_THREADS"] = "16"
    mitsuba.set_variant('llvm_ad_mono')
    data_path = Path.cwd() / "moose/input/circular_glass_out.e"

    output_dir = Path.cwd() / "validation/mitsuba/output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Camera parameters (matching your Blender setup)
    cam_pixels = np.array([2464, 2056])
    cam_pixel_size = np.array([0.00345, 0.00345])  # mm
    cam_position = np.array([0, 0, 400])  # mm
    focal_length = 35.0  # mm

    # Calculate FOV from focal length and sensor size
    sensor_width = cam_pixels[0] * cam_pixel_size[0]
    sensor_height = cam_pixels[1] * cam_pixel_size[1]
    fov_x = 2 * np.arctan(sensor_width / (2 * focal_length)) * 180 / np.pi
    print(f"{fov_x=}")
    fov_y = 2 * np.arctan(sensor_height / (2 * focal_length)) * 180 / np.pi

    # Create checkerboard texture
    rows, cols = 80, 90
    square_size = 5
    n_rows = rows // square_size
    n_cols = cols // square_size
    checkerboard_single = np.indices((n_rows, n_cols)).sum(axis=0) % 2
    checkerboard = np.repeat(np.repeat(checkerboard_single, square_size, axis=0), 
                            square_size, axis=1)
    checkerboard_rgb = np.stack([checkerboard] * 3, axis=-1).astype(np.float32)

    # Save checkerboard as bitmap for Mitsuba
    plt.imsave(output_dir / 'checkerboard.png', checkerboard_rgb, cmap='gray')

    deformed_mesh_file = output_dir / "deformed_window.ply"
    export_moose_to_mitsuba(data_path, deformed_mesh_file)

    light_intensity = (5.0*10**6) / (4*np.pi * 2)

    # Build Mitsuba scene
    scene_dict = {
        'type': 'scene',
        
        # Integrator (path tracer)
        'integrator': {
            'type': 'photomapper',
            'max_depth': 8, 
        },
        
        # Camera
        'camera': {
            'type': 'perspective',
            'fov': fov_x,
            'fov_axis': 'x',
            'to_world': mitsuba.ScalarTransform4f.look_at(
                origin=[0, 0, 400],    # Camera position (mm)
                target=[0, 0, 0],       # Look at origin
                up=[0, 1, 0]         
            ),
            'film': {
                'type': 'hdrfilm',
                'width': cam_pixels[0],
                'height': cam_pixels[1],
                'rfilter': {
                    'type': 'box'  # Sharp pixel filter
                }
            },
            'sampler': {
                'type': 'independent',
                'sample_count': 800  # Samples per pixel 
            }
        },

            'target': {
            'type': 'rectangle',
            'to_world': mitsuba.ScalarTransform4f.scale([45, 40, 1]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'bitmap',
                    'filename': str(output_dir / 'checkerboard.png'),
                    'filter_type': 'nearest',
            }
        }
    },
        
        'window': {
            'type': 'ply',
            'filename': str(deformed_mesh_file),
            'to_world': mitsuba.ScalarTransform4f.translate([0, 0, 125]), # Already in mm position from MOOSE
            'bsdf': {
                'type': 'dielectric',
                'int_ior': 1.3777, # MgF2
            }
        },
        
        'light': {
            'type': 'point',
            'position': [200, 0, 200],
            'intensity': {
                'type': 'rgb',
                'value': [light_intensity, light_intensity, light_intensity]  
            }
        },
       
        'background': {
        'type': 'constant',
        'radiance': {
            'type': 'rgb',
            'value': [0.5, 0.5, 0.5] 
        }
    }
    }

    # Create scene
    scene = mitsuba.load_dict(scene_dict)

    # Render the image
    print("Rendering Mitsuba scene...")
    image = mitsuba.render(scene)  # samples per pixel

    # Save rendered image
    output_png = output_dir / 'mitsuba_render_0_big.png'
    bmp = mitsuba.util.convert_to_bitmap(image, uint8_srgb=True)
    mitsuba.util.write_bitmap(str(output_png), bmp)
    print(f"Saved PNG` to: {output_png}")

if __name__ == "__main__":
    main()
