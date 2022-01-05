#Script to sample data from the dataset for use with the tagger.
#Author: Michael Curcio

import sys
import os
import random
import shutil

rootDir = sys.argv[1]
sampleSize = sys.argv[2]
outDir = sys.argv[3]

fileNames = []

for root, dirs, files in os.walk(rootDir):
    for name in files:
        if name[-4:] == ".png":
            fileNames.append(os.path.join(root,name))

numFiles = len(fileNames)
ndx = random.sample(range(1,numFiles), int(sampleSize))

randomSample = [fileNames[i] for i in ndx]

i=0
#We rename the 5th to last character, this is where the numbers used to be,
# we give them unique names.
for name in randomSample:
    newName = "sampled-screenshot_" + str(i) + ".png"
    shutil.copy2(name, outDir)

    dstFile = os.path.join(outDir, name)
    newFile = os.path.join(outDir, newName)
    os.rename(dstFile, newFile)
    i = i + 1

#print(randomSample)

