#Script to convert a h5 keras file into a tensorflow checkpoint


import os # for getting number of files in a directory via os.listdir()
import sys # for getting command line arguments

if len(os.sys.argv) < 2:
    print("Converts a h5 keras file into a tensorflow checkpoint.")
    print("\nusage: python " + __file__ + " <.h5 file>")
    print("i.e. " + "python " + __file__ + " InceptionV3.h5\n")
    print("***Note: make sure you are using python 3.6 with the LATEST VERSIONS of keras-gpu and tensorflow-gpu backends.\n")
    exit()

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, GlobalAveragePooling2D

from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D



from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator


from math import ceil # for ceil

import keras


import tensorflow as tf

h5_filename = os.sys.argv[1]

num_files_wd = len(os.listdir(".")) #number of files in the working directory
base_filename = "InceptionV3-" + str(num_files_wd)
ckpt_filename = base_filename + ".ckpt"

model = keras.models.load_model(h5_filename)

print("Converting " + h5_filename + " to a tensorflow checkpoint...")
# load the saved model and convert it to a tensorflow ckpt file (keras cant do this itself)
saver = tf.train.Saver()

sess = keras.backend.get_session()
save_path = saver.save(sess, "./" + ckpt_filename)
print("Successfully saved tensorflow checkpoint as " + ckpt_filename + ".")
