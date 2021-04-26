#!/usr/bin/env python

import uproot
import uproot_methods
import pandas as pd
import numpy as np
import os, time, json
import sys

def _run():

    infile = "LongPol"
    ROOTfile = uproot.open(infile+".root")
    tree=ROOTfile["UFO"]

    branches = tree.arrays(namedecode='utf-8')
    df = pd.DataFrame(branches)
    

    label_list = ['truth_isW', 'truth_isZ', 'truth_isLong', 'truth_isTrans']
    labels = df[label_list].values
    
    nconst = 100
    df = df[df.columns[17:17+8*nconst]].values.reshape(-1,nconst,8)

    #____________________________HERE WE HAVE A DATASET shaped (njet,nconst,nfeatures=8)______________________
    #________________________________the nfeatures are in the following order_________________________________
    #______(delta_eta)::(delta_phi)::(delta_R)::(log_pt)::(log_E)::(log_pt_frac)::(log_E_frac)::(taste)_______
    
    #WE CAN DECIDE TO MASK IF EVERYTHING IS -999
    #mask = (df == [-999,-999,-999,-999,-999,-999,-999,-999])
    #OR WE CAN DECIDE TO MASK IF PT HAS A SMALL VALUE FOR INSTANCE
    mask = (df[:,:,3]==0) #to be tested, look at pt values
    mask=np.array([maskpt,]*8).transpose(1,2,0)
    df = np.ma.masked_array(threeD,mask=maskpt, fill_value=np.nan)
    #THIS IS NEEDED DUE TO A SMALL MISTAKE ON MY PART WHEN CREATING THE NTUPLES
    #WHEN COMPUTING THE RATIOS THE COMP PT AND E WERE IN MeV and THE JETS' WERE IN GeV
    df[:,:,5:7] -= np.log(1e3)
   
    jets=df.filled()
    jets_reshaped = jets.reshape(jets.shape[0], -1)
    np.savetxt(infile+'_features.csv', jets_reshaped)
    np.savetxt(infile+'_labels.csv', labels)

    print("Finished")

#_______________________WHAT I USED TO DO BEFORE I USED FLAT NTUPLES______________________________________
#    part_eta = df['comp_eta'].values
#    part_phi = df['comp_phi'].values
#    part_pt = df['comp_pt'].values
#    part_e = df['comp_E'].values
#    part_taste = df['comp_taste'].values
#    part_eta = np.stack(part_eta)
#    part_phi = np.stack(part_phi)
#    part_pt = np.stack(part_pt)
#    part_e = np.stack(part_e)
#    part_taste = np.stack(part_taste)

#    twoD = np.concatenate((part_eta,part_phi,part_pt,part_e,part_taste), axis=1)
#    threeD = twoD.reshape(-1, 5, 100).transpose(0,2,1)
#    maskpt = (threeD[:,:,2]==0)
#    maskpt=np.array([maskpt,]*5).transpose(1,2,0)
#    threeD = np.ma.masked_array(threeD,mask=maskpt, fill_value=np.nan)

#    fracs = np.concatenate([(threeD[:,:,2]/ jet_pt.reshape(-1,1)).reshape(-1,100,1), (threeD[:,:,3]/ jet_e.reshape(-1,1)).reshape(-1,100,1)],axis=-1)
#    fracs = np.ma.masked_array(fracs,mask=threeD.mask[:,:,:2], fill_value=np.nan)

#    diffs = np.concatenate([part_eta_rel.reshape(-1,100,1), part_phi_rel.reshape(-1,100,1)],axis=-1)
#    diffs = np.ma.masked_array(diffs,mask=threeD.mask[:,:,:2], fill_value=np.nan)
#    total = np.ma.concatenate([fracs,diffs],axis=-1)
#    total = np.dstack((total,threeD[:,:,4]))
 

if __name__ == "__main__":
    _run()

