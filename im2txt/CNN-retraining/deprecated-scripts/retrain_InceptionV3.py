#Script to transfer learn, fine tune, or learn from scratch for Inception V3 from a given custom data split
#This saves a h5 keras file AND a tensorflow checkpoint 

import os # for getting number of files in a directory via os.listdir()
import sys # for getting command line arguments

if len(os.sys.argv) < 5:
    print("Retrains an InceptionV3 model either by transfer learning (changing just the last layer), fine tuning (changing last layer and layers deeper into the model), or by learning from scratch (initializing the InceptionV3 architecture with random weights).")
    print("\nusage: python " + __file__ + " <mode; 0 = transfer learning, 1 = fine tuning, 2 = learning from scratch> <epochs> <number of gpus to train on> <directory with 'train', 'val', and 'test' subdirectories>")
    print("i.e. " + "python " + __file__ + " 0 3 1 ./data\n")
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

num_gpus = int(os.sys.argv[3]) # number of gpus to train on

if num_gpus <= 0:
    print("Error: num_gpus must be >= 1")
    exit()

batch_size = 64 * (num_gpus) #number of images per batch (constant)

mode = int(os.sys.argv[1])

num_epochs = int(os.sys.argv[2]) #number of epochs to train for





if ((mode != 0) and (mode != 1) and (mode != 2)):
    print("error: mode must be 0, 1, or 2")
    exit()


fine_tune = False # whether to adjust weights deep into the model
from_scratch = False # whether to train the model from scratch; if this is false, the imagenet weights are used instead



if mode == 0: # transfer learning (readjust just the weights at the final layer of the model for classification)
    print("Preparing to transfer learn...\n")
    fine_tune = False
    from_scratch = False
elif mode == 1: # fine tuning (readjust weights deep into the model)
    print("Preparing to fine tune...\n")
    fine_tune = True
    from_scratch = False
elif mode == 2: # learning from scratch
    print("Preparing to learn from scratch...\n")
    fine_tune = True
    from_scratch = True

data_dir = os.sys.argv[4]


if (not os.path.isdir(data_dir)):
    print("error: '" + data_dir + "' does not exist. Exiting.")
    exit()
    


train_dir = os.path.join(data_dir, "train")   #location of train directory
val_dir = os.path.join(data_dir, "val")       #location of validation directory
test_dir = os.path.join(data_dir, "test")     #location of test directory



if (not os.path.isdir(train_dir)):
    print("error: '" + train_dir + "' does not exist. Exiting.")
    exit()


if (not os.path.isdir(val_dir)):
    print("error: '" + val_dir + "' does not exist. Exiting.")
    exit()

num_trainimgs = 0 # number of training images in dataset
num_valimgs = 0 # number of validation images in dataset
num_testimgs = 0 #number of test images in dataset

train_categories = os.listdir(train_dir)
val_categories = os.listdir(val_dir)
test_categories = os.listdir(test_dir)

num_classes = len(os.listdir(train_dir)) #number of classes to train on (constant)


# get number of train images

for direc in train_categories:
    num_trainimgs += len(os.listdir(os.path.join(train_dir, direc))) # add the number of files in this folder to num_trainimgs
    


# get number of validation images

for direc in val_categories:
    num_valimgs += len(os.listdir(os.path.join(val_dir, direc))) # add the number of files in this folder to num_valimgs



# get number of test images

for direc in test_categories:
    num_testimgs += len(os.listdir(os.path.join(test_dir, direc))) # add the number of files in this folder to num_valimgs



print(str(num_trainimgs) + " training images across " + str(len(train_categories)) + " categories.")
print(str(num_valimgs) + " validation images across " + str(len(val_categories)) + " categories.")
print(str(num_testimgs) + " test images across " + str(len(test_categories)) + " categories.")


print("Split: " + str(int(((num_trainimgs)/(num_trainimgs + num_valimgs)) * 100)) + "% training, " + str(int(((num_valimgs)/(num_trainimgs + num_valimgs)) * 100)) + "% validation")
print("\n")
assert( ( (num_classes == len(train_categories)) and (num_classes == len(val_categories)) ) ) # make sure num_classes is equal to the number of folders in the train and val directories


train_steps = ceil(num_trainimgs / batch_size) #steps per epoch

val_steps = ceil(num_valimgs / batch_size)


##############################################  END OF CONSTANTS  ##############################################  



model = Sequential()

input_layer = Input((image_size, image_size, 3))


model.add(InceptionV3(weights=(None if from_scratch else 'imagenet'), input_tensor=input_layer, include_top=False)) # initialize inception v3 with imagenet weights; use None for randomly initialized weights (from scratch)
	                                           
                                                                                                                    # include_top = False gets rid of the last layer usually used for classification
model.add(Flatten()) #To flatten 4D into 2D                                                                         # (so we can sort into our own categories)

model.add(Dense(num_classes, activation='softmax'))



# Say not to train first layer (InceptionV3) model. It is already trained
#Note that layers[0] is the InceptionV3 model because of the first model.add
model.layers[0].trainable = fine_tune #True for fine tuning, False for transfer learning
                                         #Fine tuning = readjust weights deep into the model,
                                         #Transfer learning = readjust just the weights at the final layer for classification


print("Model structure: ", model.summary())
#print("Model weights: ", model.get_weights())




if num_gpus >= 2: # i.e. if we are doing more than one gpu, construct a multi gpu model
    model = multi_gpu_model(model, gpus=num_gpus) # Construct a copy of model on 3 gpus for distributed gpu training
    


#look into optimizer objects for better performance
model.compile(optimizer='adam', #stochastic gradient descent
                     loss='categorical_crossentropy', #i.e log loss 
                     metrics=['accuracy']) #report an accuracy metric (what fraction of our predictions are correct)




#ImageDataGenerator (populate training and validation data)

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input) #preprocess input puts all numbers from an input nmpy array into range [-1,1]



#create train generator
train_generator = data_generator.flow_from_directory(train_dir, target_size=(image_size, image_size), batch_size=batch_size, class_mode='categorical')

#create validation generator
validation_generator = data_generator.flow_from_directory(val_dir, batch_size=batch_size, target_size=(image_size, image_size), class_mode='categorical')

test_generator = data_generator.flow_from_directory(test_dir, batch_size=batch_size, target_size=(image_size, image_size), class_mode='categorical')

#train model
model.fit_generator(train_generator, steps_per_epoch=train_steps, epochs = num_epochs, validation_data=validation_generator, validation_steps=val_steps)

print("\n*****************************************************************\n\nEnd of training. Now evaluating on the test set...\n\n*****************************************************************\n\n")



#evaluate the model


result = model.evaluate_generator(test_generator, steps = num_testimgs//batch_size)

print("Result of evaluating the model: " + str(result))



print("Done evaluating! Now saving...")
#save model in h5 format

num_files_wd = len(os.listdir(".")) #number of files in the working directory

base_filename = "InceptionV3-" + str(num_files_wd)
h5_filename = base_filename + ".h5"
ckpt_filename = base_filename + ".ckpt"
model.save(h5_filename)
print("Successfully saved " + h5_filename + ".")


print("Converting " + h5_filename + " to a tensorflow checkpoint...")
# load the saved model and convert it to a tensorflow ckpt file (keras cant do this itself)
saver = tf.train.Saver()
#model = keras.models.load_model(h5_filename)
sess = keras.backend.get_session()
save_path = saver.save(sess, "./" + ckpt_filename)
print("Successfully saved tensorflow checkpoint as " + ckpt_filename + ".")
