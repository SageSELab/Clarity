# Script to add synthetically generated components to the training set and then sample from the perturbed data until each class has at east 5000 training examples (without overwriting)
# This assumes that the files in the input directories have the form "CheckBox-0a0b7-000138dc2-6f95275b02ac93053c8aa290df69ef9ff8db7ad6.png" (meaning that they were run with ui_counter already)

# From redraw paper:

'''
Then for the training set, we
added the synthetically generated components to the set
of organic GUI-component training images, and performed
color perturbation on only the training data (after segmen-
tation) until each class had at least 5K training examples.

'''

import os
import sys
import shutil



import random

random.seed()

if len(os.sys.argv) < 4:
    print("\nAdds synthetically generated components to the training set and then samples from the perturbed data until each class has at east 5000 training examples\n\nusage: python " + __file__ + " <training set dir> <synthetic dir> <perturbed dir>")
    print("i.e. python " + __file__ + " CNN-Evaluation/Partitioned-Organic-Data-Split/Training Synthesized-Leaf-Nodes CNN-Evaluation/Synthetic+Organic-Color-Perturbed-Training-Data-Split")
    print("")
    exit()

train_dir = os.sys.argv[1]   # CNN-Evaluation/Partitioned-Organic-Data-Split/Training
synth_dir = os.sys.argv[2]   # Synthesized-Leaf-Nodes
perturb_dir = os.sys.argv[3] # CNN-Evaluation/Synthetic+Organic-Color-Perturbed-Training-Data-Split


train_files = os.listdir(train_dir)
synth_files = os.listdir(synth_dir)
perturb_files = os.listdir(perturb_dir)




train_components = {} #dictionary of component and count
synth_components = {} #dictionary of component and count
perturb_components = {} #dictionary of component and count

#Perform an initial count of each directory

for f in train_files:
    comp = f[0:f.find("-")] #We expect files to be in the following format (as an example): "CheckBox-0a0b7-000138dc2-6f95275b02ac93053c8aa290df69ef9ff8db7ad6.png"
    
    if not comp in train_components:
        train_components[comp] = 1
    else:
        train_components[comp] += 1
            



for f in synth_files:
    comp = f[0:f.find("-")]
    
    if not comp in synth_components:
        synth_components[comp] = 1
    else:
        synth_components[comp] += 1
            
            


for f in perturb_files:
    comp = f[0:f.find("-")]
    
    if not comp in perturb_components:
        perturb_components[comp] = 1
    else:
        perturb_components[comp] += 1

print("#" * 60)
print("Initial count:\n\n")

t_sum = 0
print(str(len(train_files)) + " train files: ")
for k in train_components:
    print("     " + k + ": " + str(train_components[k]))
    t_sum += train_components[k]
print("(which sum to " + str(t_sum) +")")
print("\n\n")
    


s_sum = 0
print(str(len(synth_files)) + " synthetic files: ")
for k in synth_components:
    print("     " + k + ": " + str(synth_components[k]))
    s_sum += synth_components[k]
print("(which sum to " + str(s_sum) + ")")
print("\n\n")




p_sum = 0
print(str(len(perturb_files)) + " perturbed files: ")
for k in perturb_components:
    print("     " + k + ": " + str(perturb_components[k]))
    p_sum += perturb_components[k]
print("(which sum to " + str(p_sum) + ")")
print("\n")
print("#" * 60)
    
    

print("\nChecking for synthetic filename collisions...")
    
# Check for filename collisions between training set and synthetic set

collisions = [] # list of filepaths that collide (it's unlikely to collide but just in case)

for f in synth_files:
    if os.path.isfile(os.path.join(train_dir, f)):
        collisions.append(os.path.join(synth_dir,f))

if len(collisions) == 0: # i.e. if there are no collisions, move all the synth_files into the train_dir
    
    print("\nNo collisions found!")
    
    print("Moving all synthetic files to " + train_dir + " ...\n")
    
    for f in synth_files:
        shutil.move(os.path.join(synth_dir,f), train_dir)
        print("Moved " + os.path.join(synth_dir,f))
    
    print("Done.")
        
else:
    print("\nCollisions present, so no files are being moved: ")
    
    for file_name in collisions:
        print("Collision: " + file_name)
        
    print("Exiting...")
    exit()





train_components = {}
train_files = os.listdir(train_dir)

for f in train_files:
    comp = f[0:f.find("-")] #We expect files to be in the following format (as an example): "CheckBox-0a0b7-000138dc2-6f95275b02ac93053c8aa290df69ef9ff8db7ad6.png"
    
    if not comp in train_components:
        train_components[comp] = 1
    else:
        train_components[comp] += 1


print("\n\n\nTraining set update:\n")

t_sum = 0
print(str(len(train_files)) + " train files: ")
for k in train_components:
    print("     " + k + ": " + str(train_components[k]))
    t_sum += train_components[k]
print("(which sum to " + str(t_sum) +")")
print("\n\n")



scarce_components = {} #dictionary to hold all components in training set with less than 5000 elements

print("\n\n\nComponents in training set with less than 5000:\n")
for k in train_components:

    if train_components[k] < 5000:
        scarce_components[k] = train_components[k]
        print("     " + k + ": " + str(train_components[k]))
        

print("\nSampling from perturbed data...")

perturb_samples = {} # dictionary mapping Component : [filename1, filename2, filename3] ; i.e. "ImageButton" : ["1.png", "2.png", "3.png"]
                     # used for sampling later

for f in perturb_files:
    comp = f[0:f.find("-")]
    
    if comp in scarce_components: # we only need to sample for scarce components, the rest are fine
        if not comp in perturb_samples:
            perturb_samples[comp] = [f]
        else:
            perturb_samples[comp].append(f)


for k in perturb_samples: # for each scarce component
    #print(k + " : " + str(len(perturb_samples[k])))
    
    file_paths = perturb_samples[k]
    
    while scarce_components[k] < 5000: # while the number of this particular scarce component is less than 5000,
        
        # sample one random file from the perturbed data and remove it from the file_paths list (also move it to the training directory)
        rand_index = random.randint(0,len(file_paths) - 1)
        
        sampled_file = file_paths[rand_index] # save the file string for later
        
        if os.path.isfile(os.path.join(train_dir, sampled_file)): # if there is a collision
            print("Perturb filename collision, so not moving: " + sampled_file)
        else: # there is no filename collision
            del file_paths[rand_index] # remove it from the table so it can't be sampled again
        
            shutil.move(os.path.join(perturb_dir,sampled_file), train_dir) # move the file on the filesystem
            
            scarce_components[k] += 1 # we took sampled_file from the perturbed dataset and put it into the training dataset, reflect this in the dictionary
        


print("Done sampling from perturbed data.\n\n")

train_components = {}
train_files = os.listdir(train_dir)

print("Final training set count (" + str(len(train_files))+ " files): ")

for f in train_files:
    comp = f[0:f.find("-")] #We expect files to be in the following format (as an example): "CheckBox-0a0b7-000138dc2-6f95275b02ac93053c8aa290df69ef9ff8db7ad6.png"
    
    if not comp in train_components:
        train_components[comp] = 1
    else:
        train_components[comp] += 1

for k in train_components:
    print("     " + k + ": " + str(train_components[k]))










