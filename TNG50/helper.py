import numpy as np
import scipy.stats
import h5py
import sys
import os
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy import units as u
import matplotlib.patches as patches
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors
from scipy.interpolate import splrep, BSpline
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
from matplotlib.colors import to_hex
from scipy.spatial.transform import Rotation as R
from multiprocessing import Process, Pipe
import requests


bp_data = r"N:\TNG50"   # Mounted drive with illustris_python and data
print("Base path exists?", os.path.exists(bp_data))

headers = {"api-key":"0cb1ff1d8991de6d20fa7a1b6c5e506b"}

# set up authentication process to access illustris/TNG API
def get(path, params=None):
    # make HTTP GET request to path
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically
    return r

# defines a cosmological model based on the Lambda CDM model, assuming flat universe
from astropy.cosmology import FlatLambdaCDM

# converts snapshot data to redshift
# snapshot data is data outputted at various times through cosmic history, with the smaller the earlier
def find_redshift(snap):
    if len(snap)==1:
        arg=np.where(dataredshift1['snap']==snap)[0]
        if len(arg)>0:
            return dataredshift1['redshift'][arg[0]]
    else:
        red=np.zeros_like(snap,dtype=float)
        for i in range(len(red)):
            arg=np.where(dataredshift1['snap']==snap[i])[0]
            if len(arg)>0:
                red[i]=dataredshift1['redshift'][arg[0]]
        return red

# converts scale factor to redshift
def redshift(scale):
    return 1.0/scale - 1.0

# converts redshift to scale factor
def find_scale(redshift):
    return 1.0/(redshift+1)

cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)

def age(redshift):
    age1=cosmo.age(redshift)
    return 13.7 - age1.value

def find_rotation_angle(vector,vector0):
    vector=vector/np.linalg.norm(vector)
    vector0=vector0/np.linalg.norm(vector0)
    dotproduct=np.dot(vector,vector0)
    return np.fabs(dotproduct)


#return x,y,mu_x,mu_y for a certain projection
def get_xymu(iproj,xyz2,galaxy_pos):
    if iproj==0:
        x=xyz2[:,0]
        y=xyz2[:,1]
        mu_x=galaxy_pos[0]
        mu_y=galaxy_pos[1]
    elif iproj==1:
        x=xyz2[:,0]
        y=xyz2[:,2]
        mu_x=galaxy_pos[0]
        mu_y=galaxy_pos[2]
    elif iproj==2:
        x=xyz2[:,1]
        y=xyz2[:,2]
        mu_x=galaxy_pos[1]
        mu_y=galaxy_pos[2]
    else:
        assert(True==False)     #wrong iproj value
    return x,y,mu_x,mu_y

# returns median position of a set of 3D points
def get_pos(coordinates):
    center=np.median(coordinates,axis=0)
    return center


# calculates the reduced moment of inertia tensor for a 3D distribution of mass (usually dark matter)
def return_tensor(dm,masssel):
    tensor = np.zeros([3,3])
    rn= dm[:,0]**2 + dm[:,1]**2 + dm[:,2]**2 
    tensor[0,0] = np.sum((dm[:,1]*dm[:,1] + dm[:,2]*dm[:,2]) * masssel[:]/rn)
    tensor[1,1] = np.sum((dm[:,0]*dm[:,0] + dm[:,2]*dm[:,2]) * masssel[:]/rn)
    tensor[2,2] = np.sum((dm[:,0]*dm[:,0] + dm[:,1]*dm[:,1]) * masssel[:]/rn)
    tensor[0,1] = - np.sum(dm[:,0] * dm[:,1] * masssel[:]/rn)
    tensor[1,0] = tensor[0,1]
    tensor[0,2] = - np.sum(dm[:,0] * dm[:,2] * masssel[:]/rn)
    tensor[2,0] = tensor[0,2]
    tensor[1,2] = - np.sum(dm[:,1] * dm[:,2] * masssel[:]/rn)
    tensor[2,1] = tensor[1,2]
    return tensor



def find_eigenvectors(dm, masssel):
    tensor=return_tensor(dm,masssel)
    w,v=np.linalg.eig(tensor)
    indices = np.argsort(w)[::-1]
    w = w[indices]
    v = v.T[indices]
    return w, v

def find_eigenvectors_array(dm, masssel):
    ll=len(masssel)
    wlist=[]
    vlist=[]
    for i in range(100):
        mask=np.random.randint(ll,size=ll)
        tensor=return_tensor(dm[mask],masssel[mask])
        w,v=np.linalg.eig(tensor)
        indices = np.argsort(w)[::-1]
        w = w[indices]
        v = v.T[indices]

        wlist.append(w)
        vlist.append(v)
    wlist=np.array(wlist)
    vlist=np.array(vlist)
    return np.mean(wlist,axis=0), np.std(wlist,axis=0), np.mean(vlist,axis=0), np.std(vlist,axis=0)

def distance(pos1):
    #return np.sqrt(np.sum(pos1[0,:]**2 + pos1[1,:]**2 + pos1[2,:]**2,axis=1))
    if len(pos1)==3:
        return np.sqrt(pos1[0]**2 + pos1[1]**2 + pos1[2]**2)
    else:
        return np.sqrt(pos1[:,0]**2 + pos1[:,1]**2 + pos1[:,2]**2)

def get_percentiles(upto_peak):
    # Ensure there is more than one value to calculate percentiles
    if len(upto_peak) > 1:
        # Calculate rise differences
        rise_values = np.diff(upto_peak)
        # Consider only positive rises
        positive_rises = rise_values[rise_values > 0]
        if len(positive_rises) > 0:
            # Calculate percentiles
            percentiles = np.percentile(positive_rises, [0, 10, 20, 50, 100])
            return percentiles
        else:
            # No positive rises found
            return np.array([0, 0, 0, 0, 0])
    else:
        # Not enough values to calculate rise
        return np.array([0, 0, 0, 0, 0])

dataredshift1=np.genfromtxt(bp_data+'/data/redshift_TNG', dtype=[('snap','<i8'), ('redshift','<f8')])

def get_percentile_indices_and_values_old(upto_peak):
    if len(upto_peak) > 1:
        # Calculate rise differences and their indices
        rise_values = np.diff(upto_peak)
        indices = np.where(rise_values > 0)[0] + 1  # Indices of positive rises
        positive_rises = rise_values[rise_values > 0]

        if len(positive_rises) > 0:
            # Calculate percentiles of the rise values
            percentiles = np.percentile(positive_rises, [10, 20, 50, 100])

            # Find the indices corresponding to these percentiles
            percentile_indices = []
            percentile_values = []
            for percentile in percentiles:
                # Find the index of the rise value closest to the percentile
                index = np.argmin(np.abs(positive_rises - percentile))
                percentile_indices.append(indices[index])
                percentile_values.append(upto_peak[indices[index]])
            return np.array(percentile_indices), np.array(percentile_values)
        else:
            # No positive rises found
            return np.array([]), np.array([])
    else:
        # Not enough values to calculate rise
        return np.array([]), np.array([])

def get_percentile_indices_and_values(upto_peak,snum_upto_peak):
    print(upto_peak,snum_upto_peak)
    if ((len(upto_peak) > 3)&(np.amin(upto_peak)/np.amax(upto_peak)<0.9)):
        npas = np.argsort(upto_peak)
        tck_m = splrep(upto_peak[npas],snum_upto_peak[npas],s=len(upto_peak)*10)
        percentile_values = np.array([0.1,0.2,0.5,0.8,0.9])*np.amax(upto_peak)
        percentile_indices = (BSpline(*tck_m)(np.array([0.1,0.2,0.5,0.8,0.9])*speak_sub))
        toreport = (percentile_values>=np.amin(upto_peak))&(percentile_values<=np.amax(upto_peak))
        print(percentile_values,percentile_values[toreport],percentile_indices,percentile_indices[toreport])
        return percentile_indices[toreport], percentile_values[toreport]
    else:
        # Not enough values to calculate rise
        return np.array([]), np.array([])

def get_percentile_indices_and_values(upto_peak, snum_upto_peak):
    if (len(upto_peak) > 5) and (np.amin(upto_peak) / np.amax(upto_peak) < 0.9):
        # Smooth the data with a Savitzky-Golay filter
        # Make sure the window_length is less than or equal to the length of the dataset and an odd number
        window_length = min(len(upto_peak) // 2 * 2 + 1, 11)  # Ensuring it's an odd number
        if window_length >= len(upto_peak):
            window_length += -2 
        if window_length < 3:  # Ensure we have a sensible window length, i.e., at least 3
            window_length = 3
#        print(upto_peak, snum_upto_peak,window_length)
        smooth_upto_peak = savgol_filter(upto_peak, window_length=window_length, polyorder=2)
        npas = np.argsort(smooth_upto_peak)
        tck_m = splrep(smooth_upto_peak[npas], snum_upto_peak[npas], s=len(smooth_upto_peak) * 20)  # Increase smoothing factor
        speak_sub = np.amax(upto_peak)

        percentile_values = np.array([0.1, 0.2, 0.5, 0.8, 0.9]) * speak_sub
        spline = BSpline(*tck_m)
        percentile_indices = spline(percentile_values)

        toreport = (percentile_values >= np.amin(smooth_upto_peak)) & (percentile_values <= np.amax(smooth_upto_peak))

#        print(percentile_values, percentile_values[toreport], percentile_indices, percentile_indices[toreport])
        return percentile_indices[toreport], percentile_values[toreport]
    else:
        # Not enough values to calculate rise
        return np.array([]), np.array([])

def log_prob_powerlaw(alpha):
    # Enforce the prior on alpha (must be steeper than -1, or integral doesn't work)
    if not (-500 < alpha < -1):
        return -np.inf
    norm = (alpha+1)/(40**(alpha+1)-10**(alpha+1))
    prb = norm*X**alpha
    ll = np.sum(np.log(prb))
    return ll


def log_prob_powerlaw15(alpha):
    # Enforce the prior on alpha (must be steeper than -1, or integral doesn't work)
    if not (-500 < alpha < -1):
        return -np.inf
    norm = (alpha+1)/(40**(alpha+1)-15**(alpha+1))
    prb = norm*X**alpha
    ll = np.sum(np.log(prb))
    return ll

# Following https://cmps-people.ok.ubc.ca/jbobowsk/Python/html/Jupyter%20Weighted%20Linear%20Fit.html
def linearFunc(x, intercept, slope):
    y = slope*x + intercept
    return y
def PowerLaw(x, intercept, slope):
    y = intercept*(x**slope) 
    return y

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

    # axis = pol_to_cart(theta_rad, phi_rad)
    # Rview = rotation_matrix(axis, angle_rad)

    fname = f"{base_path}/cutout_{subfind_id}.hdf5"

    with h5py.File(fname, 'r') as h5f:
        data4 = h5f['PartType4']
        coords = data4['Coordinates'][:]/h0
        masses = data4['Masses'][:]*1e10/h0

    # Center galaxy
    gal_pos = np.median(coords, axis=0)
    coords -= gal_pos

    # Mask within 30 kpc for stellar inertia tensor
    r = np.sqrt(np.sum(coords**2, axis=1))
    mask = (r > 0) & (r < 30)
    from helper import find_eigenvectors  # ensure this is defined
    _, v0 = find_eigenvectors(coords[mask], masses[mask])

    # Rotate coordinates
    rotated_coords = coords @ v0.T

    # Rotate galaxy to have main progenitor stars horizontal
    rotated_coords = rotated_coords[:, [2, 1, 0]]  # swap Z ↔ X


    return rotated_coords.astype('f4')