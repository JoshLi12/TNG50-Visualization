from helper import *

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

bp_local = os.getcwd()  # Local TNG50 folder for output

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
from io import BytesIO

#from matplotlib import rcParams
# matplotlib style settings
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11


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

    
    # for i in range(1, len(mw_like)):
    inp = "y"
    count = 1
    while inp == "y":
        print("Iteration:", count)
        
        # The current index
        idx = 333426

        galaxy_output = os.path.join(bp_local, "individual_inspect", str(idx))
        os.makedirs(galaxy_output, exist_ok=True)

        theta = int(input("Theta: "))
        phi = int(input("Phi: "))
        angle = int(input("Angle: "))
        # theta = 0
        # phi = 0
        # angle = 0

        theta_rad = np.deg2rad(theta)
        phi_rad = np.deg2rad(phi)
        angle_rad = np.deg2rad(angle)

        subfindID0=idx

        a1,a2,a3=create_tags(subfindID0,basePath+'/output/')
        print('Particle numbers halo :',subfindID0,len(np.where(a1)[0]),len(np.where(a2)[0]),len(np.where(a3)[0]))
        angles = []
        # fname = galaxy_output+'/cutout_'+str(subfindID0)+'_' + str(theta) + "_" + str(phi) + '.hdf5'
        fname = galaxy_output+'/cutout_'+str(subfindID0)+'_' + str(theta) + "_" + str(phi) + "_" + str(angle) + '.hdf5'
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

        # # GRAPH 1 - STELLAR ORIGIN GRAPH
        # plt.figure(figsize=(12,3))
        # plt.subplot(141)
        # plt.xlim(-50,50)
        # plt.ylim(-50,50)

        # view_dir_world = pol_to_cart(theta_rad, phi_rad)

        # axis = view_dir_world @ v0  # direction you want to view from
        # R = rotation_matrix(axis, angle_rad)
        # newProj = coordinates_rot @ R.T

        # plt.plot(newProj[~a1,2],newProj[~a1,1],',', alpha=0.6)
        # plt.plot(newProj[a1,2],newProj[a1,1],',', alpha=0.1)

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

            # # Velocities
            # plt.subplot(142)
            # plt.xlim(-50,50)
            # plt.ylim(-50,50)
            # hb = plt.hexbin(coordinates_rot[~a1,2],coordinates_rot[~a1,0], C=velocities_rot[~a1,1], gridsize=100, reduce_C_function=np.median, cmap='viridis', alpha=0.6)
            # plt.colorbar(hb, label='Median Velocity')


            maj_35 = ((coordinates_rot[~a1,2]>30)&
                        (coordinates_rot[~a1,2]<40)&
                        (abs(coordinates_rot[~a1,0])<widthbin/2.0))
            if (np.sum(masses[~a1][maj_35]) > 1e5):
                v_35 = np.percentile(velocities_rot[~a1,1][maj_35], [16, 50, 84])
            # Work out an alternative for fitting the binned profiles. Minor axis first
            # Added 4 in quadrature to place a limit of +/-2 on the density profile error. 
            # plt.subplot(143)
            # n_min = np.histogram(coordinates_rot[~a1,0][tenforty_acc_min],bins=6,range=[10,40])
            # if (len(n_min[0][n_min[0]>0])>=2):
            #     densprof = n_min[0]/(widthbin*30/6)
            #     ddensprof = np.sqrt(n_min[0]+4)/(widthbin*30/6)
            #     a_fit, cov = curve_fit(PowerLaw, (0.5*(n_min[1][1:]+n_min[1][:-1])/25), densprof, sigma = ddensprof)
            #     alpha_acc_min = np.array([a_fit[1]-np.sqrt(cov[1,1]),a_fit[1],a_fit[1]+np.sqrt(cov[1,1])])
            #     plt.plot(np.log10(0.5*(n_min[1][1:]+n_min[1][:-1])),np.log10(densprof),'ro')
            #     plt.plot(np.log10(rplt),np.log10(a_fit[0]*(rplt/25)**a_fit[1]),'r-')
            # # Major axis (projected edge on)
            # n_maj = np.histogram(coordinates_rot[~a1,2][tenforty_acc_maj],bins=6,range=[10,40])
            # if (len(n_maj[0][n_maj[0]>0])>=2):
            #     densprof_maj = n_maj[0]/(widthbin*30/6)
            #     ddensprof_maj = np.sqrt(n_maj[0]+4)/(widthbin*30/6)
            #     a_fit_maj, cov_maj = curve_fit(PowerLaw, (0.5*(n_maj[1][1:]+n_maj[1][:-1])/25), densprof_maj, sigma = ddensprof_maj)
            #     alpha_acc_maj = np.array([a_fit_maj[1]-np.sqrt(cov_maj[1,1]),a_fit_maj[1],a_fit_maj[1]+np.sqrt(cov_maj[1,1])])
            #     plt.plot(np.log10(0.5*(n_maj[1][1:]+n_maj[1][:-1])),np.log10(densprof_maj),'go')
            #     plt.plot(np.log10(rplt),np.log10(a_fit_maj[0]*(rplt/25)**a_fit_maj[1]),'g-')
            # # Major axis (face on)
            # n_maj2 = np.histogram(np.sqrt(coordinates_rot[~a1,2][fifteenforty_acc_maj]**2+coordinates_rot[~a1,1][fifteenforty_acc_maj]**2),bins=6,range=[15,40])
            # if (len(n_maj2[0][n_maj2[0]>0])>=2):
            #     densprof_maj2 = n_maj2[0]/(2.0*np.pi*0.5*(n_maj2[1][1:]+n_maj2[1][:-1])*25/6)
            #     ddensprof_maj2 = np.sqrt(n_maj2[0]+4)/(2.0*np.pi*0.5*(n_maj2[1][1:]+n_maj2[1][:-1])*25/6)
            #     a_fit_maj2, cov_maj2 = curve_fit(PowerLaw, (0.5*(n_maj2[1][1:]+n_maj2[1][:-1])/25), densprof_maj2, sigma = ddensprof_maj2)
            #     alpha_acc_maj2 = np.array([a_fit_maj2[1]-np.sqrt(cov_maj2[1,1]),a_fit_maj2[1],a_fit_maj2[1]+np.sqrt(cov_maj2[1,1])])
            #     plt.plot(np.log10(0.5*(n_maj2[1][1:]+n_maj2[1][:-1])),np.log10(densprof_maj2),'b*')
            #     plt.plot(np.log10(rplt),np.log10(a_fit_maj2[0]*(rplt/25)**a_fit_maj2[1]),'b-')


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
        # os.makedirs(os.path.join(galaxy_output, "results"), exist_ok=True)
        # plt.savefig(galaxy_output+'/results/'+str(idx)+'_' + str(theta) + "_" + str(phi) +'.png', format='png',dpi=200)
        # plt.savefig(galaxy_output+'/results/'+str(idx)+'_' + str(theta) + "_" + str(phi) + "_" + str(angle) +'.png', format='png',dpi=200)
        print("Iteration Complete")
        inp = input("Continue? (y/n): ")



    #         print('Galno  log(M_10_40)       t_50(Gyr)           t_90(Gyr)')
    #         print(idx,np.log10(M1040_acc_min),t50_1040_acc_min,t90_1040_acc_min)
    #         print('[M/H]_30              d[M/H]/dz (kpc^-1)      alpha_acc_min[0/1/2]          b/a')
    #         print(Met_acc_min,Met_acc_min_grad,alpha_acc_min,bovera)
    #         print('Major axis : log(M_15_40)    t_90(Gyr)     [M/H]_30              d[M/H]/dz (kpc^-1)      alpha_acc_maj')
    #         print(np.log10(M1540_acc_maj),t90_1540_acc_maj,Met_acc_maj,Met_acc_maj_grad,alpha_acc_maj2[1])
    #         print('v_35 (16/50/84) ')
    #         print(v_35)
    #     st;op=True



    # st;op=True