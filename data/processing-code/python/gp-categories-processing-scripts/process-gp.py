# script to read in a directory in which each file is a google play category, and each file's contents is a list of package names that belong to the file name's category
# and then output a tar file with the directory structure needed for keras:
# note: landscape screens are excluded from sampling, and only screens from the clarity dataset are considered

# out.tar:
#   test
#       COMICS
#           file1, file2, file3
#       TOOLS
#           file1, file2, file3
#       FAMILY_CREATE
#           file1, file2, file3
#       BUSINESS
#          file1, file2, file3
#       ...

#   val
#       COMICS
#           file1, file2, file3
#       TOOLS
#           file1, file2, file3
#       FAMILY_CREATE
#           file1, file2, file3
#       BUSINESS
#          file1, file2, file3
#       ...


#   train
#       COMICS
#           file1, file2, file3
#       TOOLS
#           file1, file2, file3
#       FAMILY_CREATE
#           file1, file2, file3
#       BUSINESS
#          file1, file2, file3
#       ...



# Note: the tar file contains symlinks to the actual jpeg files on bg9, so that needless copying is not needed


# example of a file:

# filename: COMICS.txt

# contents: 
#           learn.to.draw.glow.cartoon
#           baby.com.FunToyzCollector
#           com.naver.linewebtoon
#           baby.com.CKNToys
#           com.iconology.comics
#           co.pamobile.pokemon.cardmaker
#           baby.com.BabyBigMouth
#           baby.com.ToysToSee
#           com.loudcrow.marvelavengers
#           com.energysh.drawshow
#           com.zalivka.animation2
#           com.crunchyroll.crmanga
#           com.marvel.unlimited
#           com.marvel.comics
#           baby.com.KidsSurpriseFun
#           co.pamobile.yugioh.cardmaker
#           br.com.escolhatecnologia.vozdonarrador
#           com.happy2.bbmanga
#           com.dccomics.comics
#           com.instaeditor.cartoonavtar
#           ...


import os
import sys
import csv # for reading in master list
import random # for doing val, test, train split
import tarfile # for tar.gz writing

random.seed(3) # make the split reproducible (as long as the script is run on an identical master list)

if len(sys.argv) < 3:
    print("Script to create directory structure needed for training keras on google play app categories as a tar file.\n")
    print("usage: python2.7 " + __file__ + " <directory containing google play category .txt files> <machine (bg9, hudson, or tower2)")
    print("\ni.e. python2.7 " + __file__ + " ./GP-Categories bg9")
    exit()
    

gp_dir = os.sys.argv[1]

if not os.path.exists(gp_dir):
    print("Error: " + gp_dir + " does not exist.")
    exit()

if not os.path.isdir(gp_dir):
    print("Error: " + gp_dir + " is not a directory.")
    exit()


machine = os.sys.argv[2].lower().strip()

if (machine != "hudson" and machine != "bg9" and machine != "tower2"):
    print("Error: invalid machine: '" + machine + "'")
    exit()
    
print("Got machine: " + machine)

############################ Constants ############################ 

GEMMA_PREFIX = "http://173.255.245.197:8080/GEMMA-CP/Clarity/" # prefix used for all filenames


CLARITY_JPG_PATH = ""

if machine == "bg9":
    CLARITY_JPG_PATH = "/scratch/ayachnes/Clarity-Data/ClarityJpegs" #bg9
elif machine == "tower2":
    CLARITY_JPG_PATH = "/home/ayachnes/Clarity-Data/ClarityJpegs/" #tower2
elif machine == "hudson":
    CLARITY_JPG_PATH = "/home/semeru/ClarityJpegs/" #hudson




############################ End of constants ############################ 






# Begin script


# Process google play files and build a dictionary mapping package name to category (i.e.  "com.laura.fashiondesign" -> "ART_AND_DESIGN")

gp_categories = {} # dictionary mapping package name to category (i.e.  "com.laura.fashiondesign" -> "ART_AND_DESIGN")

for category_file in os.listdir(gp_dir):
    packages = [p for p in open(os.path.join(gp_dir, category_file), "r").read().split("\n") if p != "" ] # each NON BLANK line is a package
    
    category_name = category_file[0:len(category_file) - 4]
    
    for p in packages:
        gp_categories[p] = category_name


master_list_path = "../../../master-screen-list/master-list.csv"

if not os.path.isfile(master_list_path):
    print("Error: master list file '" + master_list_path + "' is not a file.")
    exit()
    
master_list = open(master_list_path, "r")

master_lines = master_list.read().splitlines()

master_list.close()

del master_lines[0] #get rid of the header row (first row)


gp_dataset = {} # dictionary mapping category name to a list of USED packages that belong to this category;
                # ex: "SPORTS" -> ['com.espn.droid.bracket_bound', 'com.ncaa.mmlive.app', 'com.handmark.sportcaster', ...]
                # i.e. for each category, this dictionary only contains packages that are part of the Clarity dataset AND that aren't blacklisted



for row in csv.reader(master_lines, delimiter=',', skipinitialspace=True):
    
    filename = row[0]  # of the form "http://173.255.245.197:8080/GEMMA-CP/Clarity/ah.creativecodeapps.tiempo-screens/screenshot_2.png"
    used = row[1]
    blacklisted = row[2] # whether this entry is blacklisted
    

    if used == "1" and blacklisted == "0": # i.e. if this screenshot was used but NOT blacklisted, we want to add it to the GP dataset
        
        names = filename[len(GEMMA_PREFIX):len(filename)].split("/") # splitting 'ah.creativecodeapps.tiempo-screens/screenshot_2.png' produces ['ah.creativecodeapps.tiempo-screens', 'screenshot_2.png']
                                                                     
        
        package_name = names[0].replace("-screens","") # take 'ah.creativecodeapps.tiempo-screens' from above and make it 'ah.creativecodeapps.tiempo'
        
        symlink_name = filename[len(GEMMA_PREFIX):len(filename)] # used as a key throughout the program in different dictionaries; this is NOT the final symlink name
        
        symlink_name = symlink_name[0:len(symlink_name)-4] + ".jpg" # replace the .png on the symlink with .jpg;  i.e. 'ah.creativecodeapps.tiempo/screenshot_2.jpg'



        if package_name in gp_categories:
            
            #print(package_name)
            
            category_name = gp_categories[package_name]
            
            #print(category_name)
            
            
            if category_name not in gp_dataset: # this is the first time we have come across this category in processing
                
                gp_dataset[category_name] = [0, [symlink_name]] # so make a new list for it
                
            else: # we have come across this category before, so append the filename to the already existing list associated with this category
                gp_dataset[category_name][1].append(symlink_name) # [1] is the screenlist
                
                
        else:
            print("Unknown package: " + package_name) # this should never happen
        


for category_name in gp_dataset: # assign gp_dataset[category_name][0] to the length of gp_dataset[category_name][1] (length of the screen list) so that this constant is available in splitting into train, test, and val
    gp_dataset[category_name][0] = len(gp_dataset[category_name][1])
    #print(category_name + " : " + str(gp_dataset[category_name]))
    #print("\n" * 4)



        

# do the train, val, test split


perc_test = .1 # percentage testing data
perc_val = .1  # percentage validation data
perc_train = 1 - perc_val - perc_test # percentage training data


train_split = {} # dictionary mapping category name to a list of USED packages that belong to this category; train split
val_split = {} # dictionary mapping category name to a list of USED packages that belong to this category; val split
test_split = {} # dictionary mapping category name to a list of USED packages that belong to this category; test split


for category_name in gp_dataset:
    
    
    # initialize empty lists for the train, val, and test of each category so that they can be appended into later
    
    train_split[category_name] = []
    val_split[category_name] = [] 
    test_split[category_name] = []
    
    num_screens = gp_dataset[category_name][0] # number of screens ORIGINALLY in the screens_list; this is immune to list resizing; since when we take a percentage after the sampling once we want it to still be a percentage relative to the original number
    screens_list = gp_dataset[category_name][1] # list of screens with a symlink name that are in this category (i.e. 'ah.creativecodeapps.tiempo-screenshot_2.jpg')
    
    
    # val
    
    for i in range(int(num_screens * perc_val)): # do val split as a percentage of the total number of screens (which is stored in screens_list[0])
            
        rand_index = random.randint(0, len(screens_list) - 1) # avoid screens_list[0] since this is a number
        
        rand_screen = screens_list[rand_index]
        
        val_split[category_name].append(rand_screen) # append rand_screen to its appropriate category in val_split
        
        del screens_list[rand_index]


    # test

    for i in range(int(num_screens * perc_test)): # do test split as a percentage of the total number of screens (which is stored in screens_list[0])
            
        rand_index = random.randint(0, len(screens_list) - 1) # avoid screens_list[0] since this is a number
        
        rand_screen = screens_list[rand_index]
        
        test_split[category_name].append(rand_screen) # append rand_screen to its appropriate category in test_split
        
        del screens_list[rand_index]



    # train

    for i in range(len(screens_list)): # do train split as all the REMAINING screens in screens_list
        
        train_split[category_name].append(screens_list[i]) # append rand_screen to its appropriate category in train_split






# Output the directory structure needed for training on google play and then compress it into a tar

tar_dir = "./Google-Play-Split" # make the directory for the tar
    
if not os.path.exists(tar_dir): # make the tar_dir if it doesn't exist
    os.system("mkdir " + tar_dir)

for split in ["train", "val", "test"]: # make train, val, and test dirs
    
    direc = os.path.join(tar_dir, split)
    
    if os.path.exists(direc):
        os.system("rm -R " + direc)

    os.system("mkdir " + direc)
    
    for category_name in gp_dataset: # make each category folder in train, test, and val (i.e. folders for "SPORTS", "ENTERTAINMENT", etc.)
        os.system("mkdir " + os.path.join(direc, category_name))
        
        split_dict = None # which dictionary to read from to output into the newly created directory
                          # this depends on whether split == "train", "val", or "test"
        
        if split == "train":
            split_dict = train_split
        elif split == "val":
            split_dict = val_split
        elif split == "test":
            split_dict = test_split
        
        
        for symlink in split_dict[category_name]: # create all the symlinks (i.e. populate this category under this split)
            
            source_path = os.path.join(CLARITY_JPG_PATH, symlink)
            
            #print("source: " + source_path)
            
            
            # symlink is of the form 'ah.creativecodeapps.tiempo-screens/screenshot_2.jpg'
            # so transform it to be in the correct form
            
            symlink = symlink.replace("-screens/","-")  # name for the symlink on bg9 (but with .png extension which is about to get replaced with .jpg)
        

            #print("symlink file name: " + symlink)
            os.symlink(source_path, os.path.join(direc, category_name, symlink)) # create the symlink
            

# Create tar file with symlinks for google play

tar_name = os.path.join(".", "Google-Play-Split-" + machine +".tar.gz")

with tarfile.open(tar_name, "w:gz") as tar:
    tar.add(tar_dir, arcname=os.path.basename(tar_dir))

os.system("rm -R " + tar_dir) # remove the directory after compression happens

print("Wrote " + tar_name)
