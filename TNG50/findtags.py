import illustris_python as il
import numpy as np
import h5py



def loadIDsSubhalos(SubfindID):
    """
    For a given Subhalo at redshift z=0, return the particleIDs, the birth snapNums,subfindIDs and groupIDs of a given Subhalo
    """
    #dataPath='/home/tnguser/TNGtags/Particles-TNG-50/'
    dataPath = r"N:\TNG50\insitu_accreted\\"
    file1=dataPath+'origins_99.hdf5'
    file2 = dataPath + r"\data_99.hdf5"
    with h5py.File(file2,'r') as f1:
        if isinstance(SubfindID,(int,np.int32,np.int64)):
            sStart=f1['SubhaloStart'][SubfindID]
            sLen=f1['SubhaloLen'][SubfindID]
            if sStart !=-1 and sLen !=-1:
                particleIDs=f1['ids'][sStart:sStart+sLen]
    with h5py.File(file1,'r') as f1:
        groupIDs=f1['GroupID'][sStart:sStart+sLen]
        snapNums=f1['SnapNum'][sStart:sStart+sLen]
        subfindIDs=f1['SubfindID'][sStart:sStart+sLen]
    return (particleIDs,snapNums,subfindIDs,groupIDs)




def getMergerTrees(subfindID0, basePath,snapNum0):
    """
    Returns a tuple containing 
    a)  a tree (with a number of essential fields)
    b)  the indices within the tree containing the mpb
    c)  the indices within the tree containing the galaxies belonging to the fof halo of the mpb, but are not on the mpb. 
    d)  the indices within the tree containing the galaxies outside the fof halo of the mpb.
    """
    fields=['SubhaloID','NextProgenitorID','MainLeafProgenitorID',
            'FirstProgenitorID','LastProgenitorID','SubhaloMassType',
            'SnapNum','SubfindID','FirstSubhaloInFOFGroupID','SubhaloGrNr', 'SubhaloPos', 'Group_R_Mean200']

    tree = il.sublink.loadTree(basePath,snapNum0,subfindID0,fields=fields,onlyMPB=False)
    
    mpb=np.where(tree['SubhaloID'][:]<=tree['MainLeafProgenitorID'][0])[0]    # main progenitor branch
    epb=np.where(tree['SubhaloID'][:]> tree['MainLeafProgenitorID'][0])[0]    # external progenitor branch
    
    mask_epb_fof=np.zeros_like(epb,dtype='bool')

    # Distinguish between FoF and External halos
    for Snap0 in np.unique(tree['SnapNum'][mpb]):
        # mpb
        arg0=np.where(tree['SnapNum'][mpb]==Snap0)[0]
        mpb_subhaloID=tree['SubhaloID'][arg0]
        # epb
        arg1=np.where(tree['SnapNum'][epb]==Snap0)[0]
        pointerIDs=tree['FirstSubhaloInFOFGroupID'][epb[arg1]]
        
        arg_fof=np.where(pointerIDs==mpb_subhaloID)[0]
        arg_mask=np.where(epb[arg1][arg_fof])[0]
        mask_epb_fof[arg1[arg_fof][arg_mask]]=True
        
    return (tree, mpb, epb[mask_epb_fof], epb[~mask_epb_fof])


def create_tags(subfindID0, basePath):
    """
    Return 3 masks of the particleIDs indicating where the particles were born: mpb, fof and ext.
    
    The algorithm consists of characterizing the birth place of the stellar particles into 3 zones (mpb, fof and ext).
    We scan the snapshots going forward in time, determining in which zone the stellar particle was formed.
    
    To speed up calculations, we make effective use of: 
    a) boolean masks for each stellar particles.  (We currenlty use 4 masks, 3 for birth in each zone, and the 4th to store previously born stellar particles.).
    b) numpy in1d function in unqiue mode to reduce everything to masks. 
    c) Element wise boolean comparisons between masks. We detect new stellar particles using the **greater** operatior, 
    and we add up stellar particles using **logical_or** operator.

    Presently the algorithm is implemented using Subhalos and the merger trees. 
    It can be modified to use Groups (FoF haloes), or incorporate other rules to determine  
    """
    
    snapNum0=99
    # get merger tree of galaxy
    tree, mpb, fof_branch, ext_branch =  getMergerTrees(subfindID0, basePath, snapNum0)
    
    particleIDs, snapNums, subfindIDs,groupIDs=loadIDsSubhalos(subfindID0)
    ll=len(particleIDs)
    
    # Mask for storing and registering where the particle was born.
    particleIDs_mpb=np.zeros_like(particleIDs,dtype='bool')
    particleIDs_fof=np.zeros_like(particleIDs,dtype='bool')
    particleIDs_ext=np.zeros_like(particleIDs,dtype='bool')

    # Mask for stellar particles already born.
    particleIDs_pre=np.zeros_like(particleIDs,dtype='bool')
    
    # The present algorithm only considers Subhaloes. More complicated rules can be constructed.

    for i in range(ll):
        if (len(np.where((tree['SnapNum'][mpb]==snapNums[i]) & (tree['SubfindID'][mpb]==subfindIDs[i]))[0])==1):
            particleIDs_mpb[i]=True
        elif (len(np.where((tree['SnapNum'][fof_branch]==snapNums[i]) & (tree['SubfindID'][fof_branch]==subfindIDs[i]))[0])==1):
            particleIDs_fof[i]=True
        elif (len(np.where((tree['SnapNum'][ext_branch]==snapNums[i]) & (tree['SubfindID'][ext_branch]==subfindIDs[i]))[0])==1):
            particleIDs_ext[i]=True
    
            
    return (particleIDs_mpb, particleIDs_fof, particleIDs_ext)
        
    
