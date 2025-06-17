import os
import sys
import numpy as np
import pyvista as pv
from matplotlib.colors import Normalize
from helper import get_galaxy_coords, bp_data, get_galaxy_met
from matplotlib.cm import get_cmap
from matplotlib import colormaps
import h5py

sys.path.insert(0, bp_data + r"\code\illustris_python")
from findtags import create_tags

subfind_id = 333426 #change-able

bp_local = os.getcwd()  # Local TNG50 folder for output
dest = os.path.join(bp_local, "galaxy_render_base")

coords = get_galaxy_coords(
    base_path=dest,
    subfind_id=subfind_id,
).astype('f4')

print(f"Loaded {len(coords)} stellar particles for SubfindID {subfind_id}.")

# a1, a2, a3 = create_tags(subfind_id, bp_data + '/output/')
# branches = [
#     (a1, [1, 0, 0], 0.0),  # Red - Main Progenitor
#     (a2, [0, 1, 0], 0.0),  # Green - FoF
#     (a3, [1, 1, 1], 0.6)   # White - External (fully transparent)
# ]


# # Initialize RGBA array
# rgba_colors = np.zeros((len(coords), 4), dtype='f4')

# # Assign colors and opacity
# for mask, rgb, alpha in branches:
#     rgba_colors[mask, :3] = rgb
#     rgba_colors[mask, 3] = alpha

# coords = coords / 20.0

# # --- Create PyVista point cloud ---
# # rgba_colors = np.hstack([colors, np.full((len(colors), 1), 0.6)])  # 30% opacity

# cloud = pv.PolyData(coords)
# cloud['rgba'] = rgba_colors
cmap = colormaps['inferno']
norm, log_metallicity = get_galaxy_met(dest, subfind_id)
# rgba_colors = cmap(norm)
# rgba_colors = rgba_colors.astype('f4')  # Ensure float32

cloud = pv.PolyData(coords)
cloud['logZ'] = log_metallicity

coords = coords / 20.0
# cloud = pv.PolyData(coords)
# cloud['rgba'] = rgba_colors

plotter = pv.Plotter(window_size=(1000, 800))
plotter.set_background([0.01, 0.01, 0.05])  # RGB values (0–1 scale)


plotter.add_points(
    cloud,
    scalars='logZ',
    # rgba=True,
    cmap='inferno',
    render_points_as_spheres=True,
    point_size=2.0,
    show_scalar_bar=False

)

plotter.add_scalar_bar(
    title='[Z/Z☉] (log scale)',
    n_labels=5,
    vertical=True,
    title_font_size=14,
    label_font_size=12,
    shadow=True
)




# plotter.add_axes(
#     interactive=True,
#     line_width=2,
#     color='white',
#     x_label='X',
#     y_label='Y',
#     z_label='Z'
# )
plotter.show_axes()


# --- Set edge-on camera (X-Z plane) ---
plotter.view_yz()  # edge-on, like looking from +Y axis
plotter.enable_trackball_style()
plotter.camera.zoom(3)  # >1 zooms in, <1 zooms out
plotter.show()
