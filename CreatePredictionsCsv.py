#!/usr/bin/env python


import uproot
import uproot_methods
import os, time, json
import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
from PC_layers import EdgeConv


def _run():

    infile_jet = "TestSample_features.csv" 
    jets = np.loadtxt(infile_jet)
    jets = jets.reshape(jets.shape[0], jets.shape[1]//8, 8)
    jets = np.ma.masked_where(jets==np.nan, jets)
    
    print("Dataset loaded")

    X_test,X_mask = jets.data,jets.mask
    
    modelfile = "model_file.tf"

    fullmodel=load_model(modelfile, custom_objects={"EdgeConv": EdgeConv})
    predictions = fullmodel.predict(X_test, batch_size=1, verbose=0)
    np.savetxt('PredFiles/TransPol_pred_isZ.csv', predictionsZ)



if __name__ == "__main__":
    _run()





















