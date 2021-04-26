# PointNet
Code for a tagger using jet constituents with a PointNet architecture. 
- PC_dir contains help functions used to create file names and call arguments.
- PC_train.py is where the training happens, while PC_layers.py is where the EdgeConv layer is defined. PC_train takes in a labels file and a features file and can also return predictions if -intest files are provided.
- ConverttoCsv.py is the script used to convert the root files into plain text files. 
- CreatePredictionsCsv.py calls the saved model and saves the predictions into a csv file. 
- CompTree.h and CompTree.cxx are the scripts used to create the flat Ntuples from the xAODs. 
- To run just do 
```
./PC_train.py -intrain 33QCD_33Long_33Trans -intrainJets 33QCD_33Long_33Trans_features.csv -intestLongJets LongPol_features.csv -intestTransJets TransPol_features.csv -intrainLabel 33QCD_33Long_33Trans_labels.csv -tagobject truth_isV -ne 5 -indir /path_to_dir -bs 1 -lr 1e-5
```
This runs well within docker://tensorflow/tensorflow:latest-gpu-jupyter
