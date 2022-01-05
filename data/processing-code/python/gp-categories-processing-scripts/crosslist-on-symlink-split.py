# python script to crosslist the test-train-val split of Clarity screenshots sorted into their google play categories to check whether any screenshots appear in more than one category
# (running this script revealed that no individual screenshots appeared in more than one category; however, it was revealed that since a single package may have multiple screenshots,
# there may be multiple datapoints that are visually similar AND in the same category. This should not affect model performance)

import os


count = 0
for f in os.listdir("."):
    path = os.path.join(".",f)
    if os.path.isdir(path):
        subcount = 0
        for g in os.listdir(path):
            subcount += len(os.listdir(os.path.join(path,g)))
        count += subcount

print(str(count) + " total files.")




screens = {} # dictionary where the keys are screens we encountered so far

count = 0
for f in os.listdir("."):
    path = os.path.join(".",f)
    if os.path.isdir(path):
        for dir_name in os.listdir(path):
            for package in os.listdir(os.path.join(path,dir_name)):
                
                package = package[0:package.find("-screenshot_")] # strip away the screenshot suffix

                if package not in screens:
                    count += 1
                    screens[package] = [dir_name]
                else:
                    if dir_name not in screens[package]:
                        screens[package].append(dir_name)

#print(screens)

dup = 0
for key in screens:
    if len(screens[key]) > 1:
        dup += len(screens[key]) -1  #print(key + " : " + str(screens[key]))

print(str(count) + " unique")
print(str(dup) + " duplicates")
