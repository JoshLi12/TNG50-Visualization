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

bp_data = r"N:\TNG50"   # Mounted drive with illustris_python and data
bp_local = os.getcwd()  # Local TNG50 folder for output
print("Base path exists?", os.path.exists(bp_data))
print("Code folder exists?", os.path.exists(os.path.join(bp_data, "code")))


sys.path.insert(0, bp_data + r"\code\illustris_python")
#sys.path.insert(0,"/Users/lsa-ericbell1/TNG50/code/")
import illustris_python as il
from scipy.interpolate import interp1d
from findtags import create_tags, getMergerTrees
from astropy.table import Table
import emcee
import corner
from scipy.linalg import lstsq
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Following https://github.com/sbi-dev/sbi/blob/main/tutorials/00_getting_started_flexible.ipynb
# https://github.com/sbi-dev/sbi/blob/main/examples/00_HH_simulator.ipynb
import torch
from sbi.inference import SNPE, simulate_for_sbi
from sbi import utils as utils
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
from sbi.analysis import pairplot
from sklearn.model_selection import train_test_split
from astropy.io import ascii
import requests
from io import BytesIO

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

#from matplotlib import rcParams
# matplotlib style settings
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11

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
def pol2Cart(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z

def rotation_matrix(axis, angle):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c

    return np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C,     y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C    ]
    ])

if __name__ == '__main__':
    basePath = bp_data#+'output'
    h0=0.6774
    G0=4.3e-6     # in units of kpc Msun^-1 (km/s)^2


    big_prog=np.load(bp_data+'/data/ans_Mstar10.0-11.3.TNG50.npy')
    mw_like = np.unique(big_prog['SubfindID'])
    Mdom_merged = np.full(len(np.hstack(mw_like)),-99.0)
    Mdom_all = np.full(len(np.hstack(mw_like)),-99.0)
    merged_all = np.full(len(np.hstack(mw_like)),True)
    snapdom_merged = np.full(len(np.hstack(mw_like)),-99)
    snapdom_all = np.full(len(np.hstack(mw_like)),-99)
    for i,idx in enumerate(np.hstack(mw_like)):
        gg = (big_prog['SubfindID']==idx)
        if len(big_prog['SubfindID'][gg])>0: 
            biggest = np.argmax(big_prog['speak'][gg])
            Mdom_all[i] = big_prog['speak'][gg][biggest]
            snapdom_all[i] = big_prog['SnapArrival'][gg][biggest]
            merged_all[i] = big_prog['Merg'][gg][biggest]
        gg = ((big_prog['SubfindID']==idx)&(big_prog['Merg']==True))
        if len(big_prog['SubfindID'][gg])>0: 
            biggest = np.argmax(big_prog['speak'][gg])
            Mdom_merged[i] = big_prog['speak'][gg][biggest]
            snapdom_merged[i] = big_prog['SnapArrival'][gg][biggest]
#        if Mdom_merged[i]==Mdom_all[i]:
#            print(idx, np.log10(Mdom_all[i]),snapdom_all[i],snapdom_merged[i],merged_all[i],'merged')
#        else:
#            print(idx, np.log10(Mdom_all[i]),snapdom_all[i],snapdom_merged[i],merged_all[i],'not yet merged')

    tdom_all = age(find_redshift(snapdom_all))
    tdom_merged = age(find_redshift(snapdom_merged))

    ########################################################################
    # Make observational quantities from snapshot data
    # from Inferring merger quantities from observables notebook
    ########################################################################

    # settings
    widthbin = 10.0 #kpc (width of HST or JWST field at D=10Mpc)
    plus = 1.0 # do I look at the north or south side of the galaxy for my profile?
    metsun = 0.0127 # TNG tells me to do this to get to solar metallicity.
    nwalkers=32
    sampler_D = emcee.EnsembleSampler(nwalkers,1,log_prob_powerlaw)
    sampler_D_maj = emcee.EnsembleSampler(nwalkers,1,log_prob_powerlaw15)

    rds = np.array([15,25,35])
    rds_m = np.array([20,30,40])


    # Primary loop, modified to start from the last known index
    # for i in range(1, len(mw_like)):
        # print("Iteration:", i)
        # status = input("Continue? (y/n): ")
        # if status == "n":
        #     break
        # The current index
    idx = 333426

    galaxy_output = os.path.join(bp_local, "individual_inspect", str(idx))
    os.makedirs(galaxy_output, exist_ok=True)

    theta = 90 # degrees from Z / changes left and right angles
    phi = 45 # degrees from X / changes up and down angles
    angle_degrees = 180 # changes rotation around axis defined by theta and phi
    angle = np.deg2rad(angle_degrees)

    subfindID0=idx
    a1,a2,a3=create_tags(subfindID0,basePath+'/output/')
    print('Particle numbers halo :',subfindID0,len(np.where(a1)[0]),len(np.where(a2)[0]),len(np.where(a3)[0]))
    angles = []
    fname = galaxy_output+'/cutout_'+str(subfindID0)+'_' + str(theta) + "_" + str(phi) + "_" + str(angle_degrees) + '.hdf5'
    if os.path.isfile(fname):
        pass
    else:
        sub_prog_url = "http://www.tng-project.org/api/TNG50-1/snapshots/%d/subhalos/%d/"%(99,subfindID0)
        cutout_request = {'dm':'all','stars':'all'}
        cutout = get(sub_prog_url+"cutout.hdf5", cutout_request)
        bytesio_object = BytesIO(cutout.content)
        with open(fname, "wb") as f:
            f.write(bytesio_object.getbuffer())
    # Read some of the data
    h1=h5py.File(fname,'r')
    data1=h1['PartType1']
    data4=h1['PartType4']
    coordinates_dm= data1['Coordinates'][:]/h0
    velocities_dm=data1['Velocities'][:]
    dm_mass=np.ones(coordinates_dm.shape[0])*3.07367708626464e05/h0
    ids_dm= data1['ParticleIDs'][:]

    coordinates= data4['Coordinates'][:]/h0
    velocities=data4['Velocities'][:]
    masses=data4['Masses'][:]*1e10/h0
    ids= data4['ParticleIDs'][:]
    str_a = data4['GFM_StellarFormationTime'][:]
    str_age = age(redshift(np.array(str_a)))

    # calculate in a simple way, the center in phase space
    gal_pos1=np.median(coordinates,axis=0)
    gal_vel1=np.median(velocities,axis=0)
    coordinates=coordinates-gal_pos1
    coordinates_dm=coordinates_dm-gal_pos1

    velocities=velocities-gal_vel1
    velocities_dm=velocities_dm-gal_vel1

    # Find the eigenvectors of the moment of inertia tensor, but restrict it to 30 kpc.
    radmax=np.sqrt(np.sum(coordinates*coordinates,axis=1))
    mask1 = (radmax > 0) & (radmax < 30)
    w,v0=find_eigenvectors(coordinates[mask1],masses[mask1])
    # Find the eigenvalues of the moment of the DM inertia tensor, within 50kpc
    # And use this to calculate c/a of the DM
    dmradmax=np.sqrt(np.sum(coordinates_dm*coordinates_dm,axis=1))
    mask1dm = (dmradmax > 0) & (dmradmax < 50)
    dmw,dmv0=find_eigenvectors(coordinates_dm[mask1dm],dm_mass[mask1dm])
    dmca = dmw[2]/dmw[0]

    coordinates_rot=np.zeros_like(coordinates)
    velocities_rot=np.zeros_like(velocities)
    coordinates_dm_rot=np.zeros_like(coordinates_dm)
    velocities_dm_rot=np.zeros_like(velocities_dm)
    for j in range(len(coordinates)):
        coordinates_rot[j,:]=np.inner(coordinates[j,:],v0)
        velocities_rot[j,:]=np.inner(velocities[j,:],v0)
    for j in range(len(coordinates_dm)):
        coordinates_dm_rot[j,:]=np.inner(coordinates_dm[j,:],v0)
        velocities_dm_rot[j,:]=np.inner(velocities_dm[j,:],v0)
    
    # GRAPH 1 - STELLAR ORIGIN GRAPH
    plt.figure(figsize=(12,3))
    plt.subplot(141)
    plt.xlim(-50,50)
    plt.ylim(-50,50)

    thetaRad = np.deg2rad(theta)
    phiRad = np.deg2rad(phi)

    x, y, z = pol2Cart(thetaRad, phiRad)
    axis = np.array([x, y, z])

    R = rotation_matrix(axis, angle)
    newProj = coordinates_rot @ R.T

    plt.plot(newProj[~a1,2],newProj[~a1,1],',', alpha=0.6)
    plt.plot(newProj[a1,2],newProj[a1,1],',', alpha=0.1)
    
    # Work out 10-40 minor axis profile
    # minor axis is either plus y or minus y with this rotated coordinate system
    # from images, it looks like 2 is the major axis, 
    # 0 and 1 are minor and intermediate axes
    # Need constrained area for metallicity, 20-40
    tenforty_acc_min = ((plus*coordinates_rot[~a1,0]>10)&
                    (plus*coordinates_rot[~a1,0]<40)&
                    (abs(coordinates_rot[~a1,2])<widthbin/2.0))
    tenforty_acc_maj = ((plus*coordinates_rot[~a1,2]>10)&
                    (plus*coordinates_rot[~a1,2]<40)&
                    (abs(coordinates_rot[~a1,0])<widthbin/2.0))
    fifteenforty_acc_maj = (((coordinates_rot[~a1,2]**2+coordinates_rot[~a1,1]**2)>15**2)&
                            ((coordinates_rot[~a1,2]**2+coordinates_rot[~a1,1]**2)<40**2))
    plt.plot(coordinates_rot[~a1,2][tenforty_acc_min],
            coordinates_rot[~a1,0][tenforty_acc_min],'r,', alpha=1)    
    plt.plot(coordinates_rot[~a1,2][tenforty_acc_maj],
            coordinates_rot[~a1,0][tenforty_acc_maj],'g,', alpha=1)    
    plt.plot(coordinates_rot[~a1,2][fifteenforty_acc_maj],
            coordinates_rot[~a1,0][fifteenforty_acc_maj],'b,', alpha=0.3)    
    if (np.sum(masses[~a1][tenforty_acc_min]) < 1e5):
        pass
    else:
        M1040_acc_min = np.sum(masses[~a1][tenforty_acc_min])
        M1540_acc_maj = np.sum(masses[~a1][fifteenforty_acc_maj])
        t90_1040_acc_min = np.percentile(13.7-
                    cosmo.age(1/data4['GFM_StellarFormationTime'][~a1][tenforty_acc_min]- 1).value,10)
        t90_1540_acc_maj = np.percentile(13.7-
                    cosmo.age(1/data4['GFM_StellarFormationTime'][~a1][fifteenforty_acc_maj]- 1).value,10)
        t50_1040_acc_min = np.percentile(13.7-
                    cosmo.age(1/data4['GFM_StellarFormationTime'][~a1][tenforty_acc_min]- 1).value,50)
        # Minor axis metallicity profiles
        Met_acc_15_25_35 = np.zeros(3)-99.0
        Met_acc_20_30_40_maj = np.zeros(3)-99.0
        for ii,rd in enumerate(rds): 
            twentyforty_acc_min = ((plus*coordinates_rot[~a1,0]>rd-5)&
                    (plus*coordinates_rot[~a1,0]<rd+5)&
                    (abs(coordinates_rot[~a1,2])<widthbin/2.0))
#           Mass-weighted mean
#           Met_acc_15_25_35[i,ii] = np.log10(np.sum(data4['GFM_Metallicity'][~a1][twentyforty_acc_min]*masses[~a1][twentyforty_acc_min])/(metsun*np.sum(masses[~a1][twentyforty_acc_min])))
#           Median metallicity
            if (np.sum(masses[~a1][twentyforty_acc_min]) > 1e5):
                Met_acc_15_25_35[ii] = np.median(np.log10(data4['GFM_Metallicity'][~a1][twentyforty_acc_min]/metsun))
        
        Met_acc_min = 0.5*(Met_acc_15_25_35[1]+Met_acc_15_25_35[2])
#       https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html
        M = rds[:, np.newaxis]**[0, 1]
        p, res, rnk, s = lstsq(M, Met_acc_15_25_35)    
        Met_acc_min_grad = p[1]

        # Major axis metallicity profiles
        for ii,rd in enumerate(rds_m): 
            twentyforty_acc_maj = (((coordinates_rot[~a1,2]**2+coordinates_rot[~a1,1]**2)>(rd-5)**2)&
                            ((coordinates_rot[~a1,2]**2+coordinates_rot[~a1,1]**2)<(rd+5)**2))
            if (np.sum(masses[~a1][twentyforty_acc_min]) > 1e5):
                Met_acc_20_30_40_maj[ii] = np.median(np.log10(data4['GFM_Metallicity'][~a1][twentyforty_acc_maj]/metsun))
        
        Met_acc_maj = Met_acc_20_30_40_maj[1]
#       https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html
        M = rds_m[:, np.newaxis]**[0, 1]
        pm, res, rnk, s = lstsq(M, Met_acc_20_30_40_maj)    
        Met_acc_maj_grad = pm[1]
        
        rplt = np.arange(10,41,1)

        # Velocities
        plt.subplot(142)
        plt.xlim(-50,50)
        plt.ylim(-50,50)
        hb = plt.hexbin(coordinates_rot[~a1,2],coordinates_rot[~a1,0], C=velocities_rot[~a1,1], gridsize=100, reduce_C_function=np.median, cmap='viridis', alpha=0.6)
        plt.colorbar(hb, label='Median Velocity')


        maj_35 = ((coordinates_rot[~a1,2]>30)&
                    (coordinates_rot[~a1,2]<40)&
                    (abs(coordinates_rot[~a1,0])<widthbin/2.0))
        if (np.sum(masses[~a1][maj_35]) > 1e5):
            v_35 = np.percentile(velocities_rot[~a1,1][maj_35], [16, 50, 84])
        # Work out an alternative for fitting the binned profiles. Minor axis first
        # Added 4 in quadrature to place a limit of +/-2 on the density profile error. 
        plt.subplot(143)
        n_min = np.histogram(coordinates_rot[~a1,0][tenforty_acc_min],bins=6,range=[10,40])
        if (len(n_min[0][n_min[0]>0])>=2):
            densprof = n_min[0]/(widthbin*30/6)
            ddensprof = np.sqrt(n_min[0]+4)/(widthbin*30/6)
            a_fit, cov = curve_fit(PowerLaw, (0.5*(n_min[1][1:]+n_min[1][:-1])/25), densprof, sigma = ddensprof)
            alpha_acc_min = np.array([a_fit[1]-np.sqrt(cov[1,1]),a_fit[1],a_fit[1]+np.sqrt(cov[1,1])])
            plt.plot(np.log10(0.5*(n_min[1][1:]+n_min[1][:-1])),np.log10(densprof),'ro')
            plt.plot(np.log10(rplt),np.log10(a_fit[0]*(rplt/25)**a_fit[1]),'r-')
        # Major axis (projected edge on)
        n_maj = np.histogram(coordinates_rot[~a1,2][tenforty_acc_maj],bins=6,range=[10,40])
        if (len(n_maj[0][n_maj[0]>0])>=2):
            densprof_maj = n_maj[0]/(widthbin*30/6)
            ddensprof_maj = np.sqrt(n_maj[0]+4)/(widthbin*30/6)
            a_fit_maj, cov_maj = curve_fit(PowerLaw, (0.5*(n_maj[1][1:]+n_maj[1][:-1])/25), densprof_maj, sigma = ddensprof_maj)
            alpha_acc_maj = np.array([a_fit_maj[1]-np.sqrt(cov_maj[1,1]),a_fit_maj[1],a_fit_maj[1]+np.sqrt(cov_maj[1,1])])
            plt.plot(np.log10(0.5*(n_maj[1][1:]+n_maj[1][:-1])),np.log10(densprof_maj),'go')
            plt.plot(np.log10(rplt),np.log10(a_fit_maj[0]*(rplt/25)**a_fit_maj[1]),'g-')
        # Major axis (face on)
        n_maj2 = np.histogram(np.sqrt(coordinates_rot[~a1,2][fifteenforty_acc_maj]**2+coordinates_rot[~a1,1][fifteenforty_acc_maj]**2),bins=6,range=[15,40])
        if (len(n_maj2[0][n_maj2[0]>0])>=2):
            densprof_maj2 = n_maj2[0]/(2.0*np.pi*0.5*(n_maj2[1][1:]+n_maj2[1][:-1])*25/6)
            ddensprof_maj2 = np.sqrt(n_maj2[0]+4)/(2.0*np.pi*0.5*(n_maj2[1][1:]+n_maj2[1][:-1])*25/6)
            a_fit_maj2, cov_maj2 = curve_fit(PowerLaw, (0.5*(n_maj2[1][1:]+n_maj2[1][:-1])/25), densprof_maj2, sigma = ddensprof_maj2)
            alpha_acc_maj2 = np.array([a_fit_maj2[1]-np.sqrt(cov_maj2[1,1]),a_fit_maj2[1],a_fit_maj2[1]+np.sqrt(cov_maj2[1,1])])
            plt.plot(np.log10(0.5*(n_maj2[1][1:]+n_maj2[1][:-1])),np.log10(densprof_maj2),'b*')
            plt.plot(np.log10(rplt),np.log10(a_fit_maj2[0]*(rplt/25)**a_fit_maj2[1]),'b-')
        # Calculate axis ratio following Harmsen
#        avdens = 0.5*(np.log10(len(coordinates_rot[~a1,0][tenforty_acc_min])*25.0**alpha_acc_min[i,1]) + 
#                  np.log10(len(coordinates_rot[~a1,2][tenforty_acc_maj])*25.0**alpha_acc_maj[i,1]))
#        rplt = np.arange(10,41,1)
#        fmaj = interp1d(np.log10(len(coordinates_rot[~a1,2][tenforty_acc_maj])*rplt**alpha_acc_maj[i,1]), np.log10(rplt),fill_value="extrapolate")
#        fmin = interp1d(np.log10(len(coordinates_rot[~a1,0][tenforty_acc_min])*rplt**alpha_acc_min[i,1]), np.log10(rplt),fill_value="extrapolate")
#        bovera[i] = 10**fmin(avdens)/10**fmaj(avdens)
        avdens = 0.5*(np.log10(a_fit[0])+np.log10(a_fit_maj[0])) #normalized at 25
        rplt = np.arange(10,41,1)
        fmaj = interp1d(np.log10(a_fit_maj[0]*(rplt/25)**a_fit_maj[1]), np.log10(rplt),fill_value="extrapolate")
        fmin = interp1d(np.log10(a_fit[0]*(rplt/25)**a_fit[1]), np.log10(rplt),fill_value="extrapolate")
        bovera = 10**fmin(avdens)/10**fmaj(avdens)
        plt.plot([fmin(avdens),np.log10(25),fmaj(avdens)],np.zeros(3)+avdens,'k+')
        plt.subplot(144)
        plt.plot(coordinates_rot[~a1,0][tenforty_acc_min], np.log10(data4['GFM_Metallicity'][~a1][tenforty_acc_min]/metsun), 'ro')
        plt.plot(rds, Met_acc_15_25_35, 'ko',markersize=10)
        plt.plot(rds_m, Met_acc_20_30_40_maj, 'b*',markersize=10)
        plt.plot(rds, p[0]+p[1]*rds, 'k-')
        plt.plot(rds_m, pm[0]+pm[1]*rds_m, 'b-')
    plt.tight_layout()
    # os.makedirs(os.path.join(bp_local, "cutouts", "results"), exist_ok=True)
    # plt.savefig(bp_local+'/cutouts/results/'+str(idx)+'.png', format='png',dpi=200)
    os.makedirs(os.path.join(galaxy_output, "results"), exist_ok=True)
    plt.savefig(galaxy_output+'/results/'+str(idx)+'_' + str(theta) + "_" + str(phi) + "_" + str(angle_degrees) +'.png', format='png',dpi=200)
    print("Iteration Complete")


            
            # print('Galno  log(M_10_40)       t_50(Gyr)           t_90(Gyr)')
            # print(idx,np.log10(M1040_acc_min),t50_1040_acc_min,t90_1040_acc_min)
            # print('[M/H]_30              d[M/H]/dz (kpc^-1)      alpha_acc_min[0/1/2]          b/a')
            # print(Met_acc_min,Met_acc_min_grad,alpha_acc_min,bovera)
            # print('Major axis : log(M_15_40)    t_90(Gyr)     [M/H]_30              d[M/H]/dz (kpc^-1)      alpha_acc_maj')
            # print(np.log10(M1540_acc_maj),t90_1540_acc_maj,Met_acc_maj,Met_acc_maj_grad,alpha_acc_maj2[1])
            # print('v_35 (16/50/84) ')
            # print(v_35)
        # st;op=True



    # st;op=True
