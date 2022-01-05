import os
import sys

#script used to verify data that comes from the hyperparameter search
# check to make sure metric values are not exactly the same, so we know
# if we need to discard some data

dataDir = sys.argv[1]
fileList = [f for f in os.listdir(dataDir) if "key" not in f]
print('detected files:')
for f in fileList:
    print(os.path.join(dataDir, f))
print('------------------------------')

#main loop
prevArr = ""
flagged = {}
for f in fileList:
    path = os.path.join(dataDir, f)
    try:
        with open(path, 'r') as curFile:
            lineNum = 0
            for line in curFile:
                arr = line.split(',')
                valArr = arr[1:] 
                if valArr == prevArr:
                    if f in flagged.keys():
                        flagged[f].append(lineNum)
                    else:
                        flagged[f] = [lineNum]
                prevArr = valArr 
                lineNum += 1
    except IOError:
        print('caught IOError')
        continue

for key in flagged.keys():
    print(key + ": " + str(flagged[key])) 
            
