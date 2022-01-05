# Script to take the finished product of the Training, Test, and Validation splits and put them into formats that keras needs for reading in

# i.e. this script should be run after ui_counter.py and ui_sampler.py

# Here is what the split looks like before this script runs:

#       Training
#          |
# file1, file2, file3 ...


#     Validation
#         |
# file1, file2, file3 ...


#        Test
#          |
# file1, file2, file3 ... 




#  Here is what the split looks like after this script runs:

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


import os
import sys

import shutil



if len(os.sys.argv) < 2:
    print("\nTakes a directory with \"Training\", \"Validation\", and \"Test\" directories and transforms it into the keras-friendly format for test, val, and train.\n\nusage: python " + __file__ + " <directory with Training, Validation, and Test directories>")
    print("i.e. python " + __file__ + " CNN-Evaluation/Partitioned-Organic-Data-Split")
    print("")
    exit()


split_dir = os.sys.argv[1]


#make sure train, val, and test directories are not already there / make sure that there is no file there with the same name


train_dir = os.path.join(split_dir, "train")
val_dir = os.path.join(split_dir, "val")
test_dir = os.path.join(split_dir, "test")

if os.path.exists(train_dir):
    print("error: " + train_dir + " already exists.")
    exit()
    
if os.path.exists(val_dir):
    print("error: " + val_dir + " already exists.")
    exit()
    
if os.path.exists(test_dir):
    print("error: " + test_dir + " already exists.")
    exit()


# make the directories
os.mkdir(train_dir)
os.mkdir(val_dir)    
os.mkdir(test_dir)




#make sure Training, Validation, and Test directories exist

Training_dir = os.path.join(split_dir, "Training")
Validation_dir = os.path.join(split_dir, "Validation")
Test_dir = os.path.join(split_dir, "Test")

if not os.path.exists(Training_dir):
    print("error: " + Training_dir + " does not exist!")
    exit()

if not os.path.exists(Validation_dir):
    print("error: " + Validation_dir + " does not exist!")
    exit()
    
if not os.path.exists(Test_dir):
    print("error: " + Test_dir + " does not exist!")
    exit()



Training_comp = {} # dictionary mapping each component type to a list of all the file paths for this component type (Training)

Validation_comp = {} # dictionary mapping each component type to a list of all the file paths for this component type (Validation)

Test_comp = {} # dictionary mapping each component type to a list of all the file paths for this component type (Test)




for f in os.listdir(Training_dir): # build Training_comp
    comp = f[0:f.find("-")]

    if not comp in Training_comp:
        Training_comp[comp] = [f]
    else:
        Training_comp[comp].append(f)


for f in os.listdir(Validation_dir): # build Validation_comp
    comp = f[0:f.find("-")]

    if not comp in Validation_comp:
        Validation_comp[comp] = [f]
    else:
        Validation_comp[comp].append(f)


for f in os.listdir(Test_dir): # build Test_comp
    comp = f[0:f.find("-")]

    if not comp in Test_comp:
        Test_comp[comp] = [f]
    else:
        Test_comp[comp].append(f)




for comp in Training_comp: #Move the training files
    
    comp_dir = os.path.join(train_dir, comp)
    os.mkdir(comp_dir) #make the directory
    
    for f in Training_comp[comp]: #for each file in the list
        shutil.move(os.path.join(Training_dir, f), comp_dir) # file from Training/ to train/COMPONENT/
    
    

for comp in Validation_comp: #Move the validation files
    
    comp_dir = os.path.join(val_dir, comp)
    os.mkdir(comp_dir) #make the directory
    
    for f in Validation_comp[comp]: #for each file in the list
        shutil.move(os.path.join(Validation_dir, f), comp_dir) # file from Validation/ to val/COMPONENT/
        

for comp in Test_comp: #Move the test files
    
    comp_dir = os.path.join(test_dir, comp)
    os.mkdir(comp_dir) #make the directory
    
    for f in Test_comp[comp]: #for each file in the list
        shutil.move(os.path.join(Test_dir, f), comp_dir) # file from Test/ to test/COMPONENT/


print("Done splitting.")
