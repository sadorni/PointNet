'''
additional functions for the support of using Keras
'''

from __future__ import absolute_import
from __future__ import print_function
import os, argparse




def construct_training_str(addition, inFile, batch_size, nb_epoch, lrate):
    training_str = str(addition)+'_'+ str(inFile)+'_'+"_trainBS"+str(batch_size)+"_nE"+str(nb_epoch)+ "_lr"+ str(lrate*1000)

    return training_str


def construct_dir_str(addition, metric):
    name=''
    name+=str(addition)+ "_metric" + str(metric) 

    return name


def create_output_files(training_str, keras_files_dir, subdir_name, tagging):
    
    os.system("mkdir -p PCFiles/%s/" % (subdir_name))
    
    model_str = keras_files_dir+subdir_name+"/poltag_model__"+training_str+"_"+str(tagging)+".tf"
    pred_str_long = "PredFiles/"+training_str+ "_Long_" + str(tagging)+".csv"
    pred_str_trans = "PredFiles/"+training_str+ "_Trans_" + str(tagging)+".csv"
    return model_str,pred_str_long, pred_str_trans






def get_training_args():
    help_input_directory = "Directory where input files are stored"
    help_training_inFile = "Training input file in HDF5 format"
    help_validation_inFile = "Validation input file in HDF5 format"
    help_tagging_object = "Object to be tagged: W,Z, Long, Trans..."
    help_training_input_branches = "Input branches to be used for training" 
    help_batch_size = "Batch size: Set the number of jets to look before updating the weights of the NN (default: %(default)s)."
    help_nb_epoch = "Set the number of epochs to train over (default: %(default)s)."
    help_learning_rate = "Learning rate used by the optimizer (default: %(default)s)."
    help_cutoff_value = "Discriminant value to use to identify isV (default: %(default)s)."


    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)

    parser.add_argument("-indir", "--input_directory", type=str,
                        default="",
                        help=help_input_directory)
    parser.add_argument("-intrain", "--training_inFile", type=str,
                        default="",
                        help=help_training_inFile)
    parser.add_argument("-intrainJets", "--training_inFile_jets", type=str,
                        default="",
                        help=help_training_inFile)
    parser.add_argument("-intestLongJets", "--testing_Long_inFile_jets", type=str,
                        default="",
                        help=help_training_inFile)
    parser.add_argument("-intestTransJets", "--testing_Trans_inFile_jets", type=str,
                        default="",
                        help=help_training_inFile)
    parser.add_argument("-intrainLabel", "--training_inFile_label", type=str,
                        default="",
                        help=help_training_inFile)
    parser.add_argument("-tagobject", "--tagging_object",
                        type=str, default='truth_isW',
                        help=help_tagging_object)
    parser.add_argument("-ne", "--nb_epoch",
                        type=int, default=80,
                        help=help_nb_epoch)
    parser.add_argument("-lr", "--l_rate",
                        type=float, default=0.001,
                        help=help_nb_epoch)
    parser.add_argument("-bs", "--batch_size",
                        type=int, default=100,
                        help=help_batch_size)
    parser.add_argument("-metric", "--metric",
                        type=str, default="accuracy",
                        help="metric to be used for trainings (default: %(default)s).")
    parser.add_argument("-loss_func", "--loss_func", type=str,
                        default="binary_crossentropy",
                        help="loss function to be used for training")

    parser.add_argument("-p", "--patience",
                        type=int, default=50,
                        help="number of epochs witout improvement of loss before stopping")


    args = parser.parse_args()
    return args
































