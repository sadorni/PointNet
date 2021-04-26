#!/usr/bin/env python

from __future__ import print_function
import os, time, json
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateScheduler, ProgbarLogger
from GetDataset import getJetsFromFlat
from PC_layers import EdgeConv
from PC_dir import get_training_args, construct_training_str, construct_dir_str, create_output_files

METRICS = {
   "accuracy": "accuracy",
   "tp": tf.keras.metrics.TruePositives(name='tp'),
   "fp": tf.keras.metrics.FalsePositives(name='fp'),
   "tn": tf.keras.metrics.TrueNegatives(name='tn'),
   "fn": tf.keras.metrics.FalseNegatives(name='fn'),
   "precision": tf.keras.metrics.Precision(name='precision'), 
   "recall": tf.keras.metrics.Recall(name='recall'),
   "AUC": tf.keras.metrics.AUC(name='auc')
}

TAGGING = {
   "truth_isW": 0,
   "truth_isZ": 1,
   "truth_isLong": 2,
   "truth_isTrans": 3
}



def _run():
    
    args = get_training_args()

    os.system("mkdir -p PCFiles/")

    #HOW MANY CONSTITUENTS DO WE WANT TO TAKE INTO ACCOUNT
    n_const = 100
    
    infile_jet = args.input_directory+args.training_inFile_jets 
    infile_label = args.input_directory+args.training_inFile_label 
    jets = np.loadtxt(infile_jet)
    label = np.loadtxt(infile_label)
    jets = jets.reshape(jets.shape[0], jets.shape[1]//8, 8)
    jets = jets[:,:n_const,:]
    jets = np.ma.masked_where(jets==np.nan, jets)
    #IF YOU DO NOT WANT TO USE THE LOGRAITHMS
    #jets[:,:,3:7] = np.exp(jets[:,:,3:7])
    #IF YOU WANT TO REMOVE FEATURES
    #jets = np.delete(jets,np.s_[2:5],2)
    

    if args.tagging_object == "truth_isZ":
    #THIS IS DONE TO REMOVE THE Ws FROM THE DATASET
      mask = (label[:,0]!=1)
      jets = jets[mask,:]
      label = label[mask,:]
      tag_n = TAGGING.get(args.tagging_object)
      Y_train = label[:,tag_n]
    if args.tagging_object == "truth_isV":
      Y_train = np.add(label[:,0], label[:,1])
    else:
      tag_n = TAGGING.get(args.tagging_object)
      Y_train = label[:,tag_n]
 
    X_train,X_mask = jets.data,jets.mask
    print("Number of jets in the training set: "+ str(len(X_train)))

    ins = Input(shape=(100,5))
    e1 = EdgeConv(k=8,filters=8)(ins)
    e2 = EdgeConv(k=8,filters=16)(e1)
    f1 = Flatten()(e2)
    d1 = Dense(units=200,activation='relu')(f1)
    d2 = Dense(units=100,activation='relu')(d1) 
    d3 = Dense(units=50,activation='relu')(d2)
    o1 = Dense(units=1,activation='sigmoid')(d3)

    for arg in vars(args):
        print(str(getattr(args, arg)))
    
    model = Model(inputs=ins,outputs=o1)
    print(model.summary())
    
    def lr_schedule(epoch):
      lr = args.l_rate
      if epoch > 10:
        lr *= 0.1
      elif epoch > 20:
        lr *= 0.01
      return lr
    
    opt = Adam(lr=lr_schedule(0))
    print(str(METRICS.get(args.metric)))
    model.compile(optimizer=opt,loss=args.loss_func,metrics=[METRICS.get(args.metric)])


    # Create directories with different names depending on the input arguments
    dir_str = construct_dir_str("PC", args.metric)
    final_dir = str(args.training_inFile) + '__' + dir_str

    print("Directory name: " + final_dir)
    os.system("mkdir -p {}".format("PCFiles/"+final_dir))


    training_str = construct_training_str("PC",args.training_inFile, args.batch_size, args.nb_epoch, args.l_rate)

    model_filename, pred_filename_Long, pred_filename_Trans = create_output_files( training_str,"PCFiles/",final_dir, args.tagging_object)

    print(training_str)

    #CALLBACKS ARE DEFINED BUT NOT USED YET
    early_stopping = EarlyStopping(monitor="val_loss", patience=args.patience)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    callback = [lr_scheduler]
    callback.append(early_stopping)
    

    model.fit(X_train,Y_train,batch_size=args.batch_size,epochs=args.nb_epoch,validation_split=0.2)
    # save general model:
    model.save(model_filename) 
    print("  --> saved model as:", model_filename)

    ##HERE ONE CAN RUN THE MODEL ON THE TEST FILES SO TO GET THE PREDICTIONS IMMEDIATELY
    infile_jet_long_test = args.input_directory+args.testing_Long_inFile_jets 
    infile_jet_trans_test = args.input_directory+args.testing_Trans_inFile_jets 
    jets_long_test = np.loadtxt(infile_jet_long_test)
    jets_trans_test = np.loadtxt(infile_jet_trans_test)
    jets_long_test = jets_long_test.reshape(jets_long_test.shape[0], jets_long_test.shape[1]//8, 8)
    jets_trans_test = jets_trans_test.reshape(jets_trans_test.shape[0], jets_trans_test.shape[1]//8, 8)
    jets_long_test = jets_long_test[:,:n_const,:]
    jets_trans_test = jets_trans_test[:,:n_const,:]
    jets_long_test = np.ma.masked_where(jets_long_test==np.nan, jets_long_test)
    jets_trans_test = np.ma.masked_where(jets_trans_test==np.nan, jets_trans_test)
    #TEST SAMPLES MUST BE PREPROCESSED LIKE THE TRAINING SAMPLES
    #jets_long_test[:,:,3:7] = np.exp(jets_long_test[:,:,3:7])
    #jets_trans_test[:,:,3:7] = np.exp(jets_trans_test[:,:,3:7])
    #jets_long_test = np.delete(jets_long_test,np.s_[2:5],2)
    #jets_trans_test = np.delete(jets_trans_test,np.s_[2:5],2)
    X_test_long,X_test_long_mask = jets_long_test.data,jets_long_test.mask
    X_test_trans,X_test_trans_mask = jets_trans_test.data,jets_trans_test.mask
    predictionsLong = model.predict(X_test_long, batch_size=args.batch_size, verbose=0)
    np.savetxt(pred_filename_Long, predictionsLong)
    predictionsTrans = model.predict(X_test_trans, batch_size=args.batch_size, verbose=0)
    np.savetxt(pred_filename_Trans, predictionsTrans)


if __name__ == "__main__":
    _run()
















