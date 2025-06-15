import os
import sys
import numpy as np
import pyvista as pv
from helper import get_galaxy_coords, bp_data

sys.path.insert(0, bp_data + r"\code\illustris_python")
from findtags import create_tags


subfind_id = 333426
theta = 90
phi = 0
angle = 0

bp_local = os.getcwd()  # Local TNG50 folder for output
dest = os.path.join(bp_local, "galaxy_render_base")

coords = get_galaxy_coords(
    base_path=dest,
    subfind_id=subfind_id,
    theta=theta,
    phi=phi,
    angle=angle
).astype('f4')

print(f"Loaded {len(coords)} stellar particles for SubfindID {subfind_id}.")

a1, a2, a3 = create_tags(subfind_id, bp_data + '/output/')
# colors = np.zeros((len(coords), 3), dtype='f4')
# colors[a1] = [1,1,1]  # Main Progenitor Branch
# colors[a2] = [1,1,1]  # Friends of Friends
# colors[a3] = [1,1,1]  # External Branch

r = np.linalg.norm(coords, axis=1)
r_log = np.log1p(r)  # smooth distance distribution
r_norm = (r_log - r_log.min()) / (r_log.max() - r_log.min())  # 0 = center, 1 = edge
brightness = (1.0 - r_norm)**1.5

# Invert so center is brightest


def color_map(val):
    if val > 0.95:
        return [1.0, 1.0, 1.0]      # small, very bright center
    elif val > 0.8:
        return [1.0, 0.7, 0.9]      # soft pink
    elif val > 0.6:
        return [0.8, 0.4, 1.0]      # purple
    elif val > 0.4:
        return [0.3, 0.4, 1.0]      # blue
    else:
        return [0.05, 0.05, 0.15]   # deep dark outer halo
def smooth_colormap(val):
    val = np.clip(val, 0.0, 1.0)

    # Define the key color steps
    stops = [0.0, 0.25, 0.5, 0.75, 1.0]
    colors = [
        [1.0, 1.0, 1.0],     # White
        [1.0, 0.6, 0.8],     # Pink
        [0.9, 0.3, 1.0],     # Magenta
        [0.5, 0.2, 1.0],     # Violet
        [0.2, 0.2, 1.0]      # Blue
    ]

    # Find which range val is in
    for i in range(len(stops) - 1):
        if stops[i] <= val <= stops[i+1]:
            t = (val - stops[i]) / (stops[i+1] - stops[i])
            c0 = np.array(colors[i])
            c1 = np.array(colors[i+1])
            return (1 - t) * c0 + t * c1

    return colors[-1]  # fallback
    
brightness = (1.0 - r_norm) ** 1.2

colors = np.array([color_map(1.0 - v) for v in r_norm])

coords = coords / 20.0

# --- Create PyVista point cloud ---
cloud = pv.PolyData(coords)
cloud['colors'] = colors

plotter = pv.Plotter(window_size=(1000, 800))
plotter.set_background([0.01, 0.01, 0.05])  # RGB values (0–1 scale)

alpha = np.clip(brightness, 0.05, 0.4).reshape(-1, 1)
rgba = np.hstack([colors, alpha])

brightness += np.random.normal(0, 0.01, brightness.shape)
brightness = np.clip(brightness, 0, 1)

# rgba_colors = np.hstack([colors, np.full((len(colors), 1), 0.6)])  # 30% opacity
cloud['rgba'] = rgba

plotter.add_points(
    cloud,
    scalars='rgba',
    rgba=True,
    render_points_as_spheres=True,
    point_size=2.0
)

print("brightness range:", np.min(brightness), "to", np.max(brightness))


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
