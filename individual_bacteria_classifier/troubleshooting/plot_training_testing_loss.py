
import numpy as np
from skimage.transform import resize
from matplotlib import pyplot as plt
from individual_bacteria_classifier.build_network_3dcnn import cnn_3d
import tensorflow as tf
from time import time
import glob
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import classification_report
import random
from sklearn.metrics import classification_report, confusion_matrix
tf.compat.v1.disable_eager_execution()


direc = glob.glob('/media/rplab/Stephen Dedalus/automated_pipeline_labels_models/data_and_labels/single_bac_labels/loss_results/training/*')

plt.figure()
for files in range(len(direc)):
    load_file = np.load(direc[files])['arr_0']
    x_array = np.arange(len(load_file))
    plt.plot(load_file, label ='validation set' + str(files))
    plt.xlabel('epoch')
    plt.legend()
    plt.ylabel('loss')


#plt.savefig('testing loss')