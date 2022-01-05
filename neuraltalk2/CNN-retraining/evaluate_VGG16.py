#This script is a standalone that will evaluate a given h5 keras model on test data (expects a "test" directory to be in the given /data directory)

import os # for getting number of files in a directory via os.listdir()
import sys # for getting command line arguments

if len(os.sys.argv) < 3:
    print("\nusage: python " + __file__ + " <data dir> <h5 file>\n")
    print("***Note: make sure you are using python 3.6 with the LATEST VERSIONS of keras-gpu and tensorflow-gpu backends.\n")
    exit()

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, GlobalAveragePooling2D

from keras.utils import multi_gpu_model #for multiple gpu action

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.models import load_model



from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator


from math import ceil # for ceil

import keras


import tensorflow as tf








# the directory with train and val should appear as:

#                            train
#    /                         |                   \
# ImageButton              ProgressBar            RadioButton ... 
#     |                        |                       |
# file1,file2,file3 ...  file1, file2, file3 ...   file1, file2, file3 ...




#                             val
#    /                         |                   \
# ImageButton              ProgressBar            RadioButton ... 
#     |                        |                       |
# file1,file2,file3 ...  file1, file2, file3 ...   file1, file2, file3 ...



#                             test
#    /                         |                   \
# ImageButton              ProgressBar            RadioButton ... 
#     |                        |                       |
# file1,file2,file3 ...  file1, file2, file3 ...   file1, file2, file3 ...





##############################################     CONSTANTS     ##############################################  


image_size = 299  #Inception V3 has 299 x 299 images as input

multi_gpu = True #whether to use multiple gpus in training or not

batch_size = 100 * (3 if multi_gpu else 1)#number of images per batch (constant)

data_dir = os.sys.argv[1]

h5 = os.sys.argv[2]

if (not os.path.isdir(data_dir)):
    print("error: '" + data_dir + "' does not exist. Exiting.")
    exit()
    


test_dir = os.path.join(data_dir, "test")       #location of test directory

num_testimgs = 0

for direc in os.listdir(test_dir):
    num_testimgs += len(os.listdir(os.path.join(test_dir, direc))) # add the number of files in this folder to num_valimgs


##############################################  END OF CONSTANTS  ##############################################  


print("Loading model from " + h5 + "...")
model = load_model(h5)

print("Model structure: ", model.summary())
#print("model weights: ", model.get_weights())


print("Turning model into a parallel model...")

parallel_model = multi_gpu_model(model, gpus=3)

print("Compiling parallel model...")

#look into optimizer objects for better performance
parallel_model.compile(optimizer='sgd', #stochastic gradient descent
                     loss='categorical_crossentropy', #i.e log loss 
                     metrics=['accuracy']) #report an accuracy metric (what fraction of our predictions are correct)

#ImageDataGenerator (populate training and validation data)


data_generator = ImageDataGenerator(preprocessing_function=preprocess_input) #preprocess input puts all numbers from an input nmpy array into range [-1,1]

print("Evaluating model...")


#test model
test_generator = data_generator.flow_from_directory(test_dir, batch_size=batch_size, target_size=(image_size, image_size), class_mode='categorical')

result = parallel_model.evaluate_generator(test_generator, steps = num_testimgs//batch_size)

print("Result of evaluating the model: " + str(result))


