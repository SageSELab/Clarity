# Script to save an inception v3 model in the old tensorflow format with imagenet weights.

import os # for getting number of files in a directory via os.listdir()


print("\nScript to save an InceptionV3 model in the old tensorflow format with imagenet weights.\n\n")
print("***Note: make sure you are using python 3.6 with the LATEST VERSIONS of keras-gpu and tensorflow-gpu backends.\n")


from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, GlobalAveragePooling2D

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions
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


image_size = 299  #Inception V3 has 299 x 299 images as input


model = Sequential()

input_layer = Input((image_size, image_size, 3))


model.add(InceptionV3(weights='imagenet', input_tensor=input_layer, include_top=True)) # initialize inception v3 with imagenet weights

#model.add(Dense(num_classes, activation='softmax'))

# Say not to train first layer (InceptionV3) model. It is already trained
#Note that layers[0] is the InceptionV3 model because of the first model.add
#model.layers[0].trainable = fine_tune #True for fine tuning, False for transfer learning
                                         #Fine tuning = readjust weights deep into the model,
                                         #Transfer learning = readjust just the weights at the final layer for classification


print("Created InceptionV3 Model.\nModel structure: ", model.summary())
#print("Model weights: ", model.get_weights()) 


#look into optimizer objects for better performance
model.compile(optimizer='adam', #stochastic gradient descent
                     loss='categorical_crossentropy', #i.e log loss 
                     metrics=['accuracy']) #report an accuracy metric (what fraction of our predictions are correct)





print("Saving InceptionV3 model...")
#save model in h5 format

num_files_wd = len(os.listdir(".")) #number of files in the working directory

base_filename = "InceptionV3-" + str(num_files_wd)
h5_filename = base_filename + ".h5"
ckpt_filename = base_filename + ".ckpt"
model.save(h5_filename)
print("Successfully saved " + h5_filename + ".")


print("Converting " + h5_filename + " to a tensorflow checkpoint...")
# load the saved model and convert it to a tensorflow ckpt file (keras cant do this itself)

saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)


sess = keras.backend.get_session()
save_path = saver.save(sess, "./" + ckpt_filename)
print("Successfully saved tensorflow checkpoint as " + ckpt_filename + ".")
