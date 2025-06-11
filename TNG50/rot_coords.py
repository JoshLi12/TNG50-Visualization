import numpy as np
import h5py


# Additional functions
def pol_to_cart(theta, phi):
    # Theta and phi must be in radians
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])

def rotation_matrix(axis, angle_rad):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    C = 1 - c

    return np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C,     y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C    ]
    ])

def get_galaxy_coords(base_path, subfind_id, h0=0.6774, theta=0, phi=0, angle=0):
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)
    angle_rad = np.deg2rad(angle)

    axis = pol_to_cart(theta_rad, phi_rad)
    Rview = rotation_matrix(axis, angle_rad)

    fname = f"{base_path}/individual_inspect/{subfind_id}/cutout_{subfind_id}.hdf5"
    with h5py.File(fname, 'r') as h5f:
        coords = h5f['PartType4']['Coordinates'][:]/h0

    coords -= np.median(coords, axis=0)

    # Optional: rotate to galaxy frame (you can skip if already aligned)
    return coords @ Rview.T