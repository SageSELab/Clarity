# python script to crosslist raw GP data check whether any packages appear in more than one category (arunning this script reveals that there are 307 packages that appear in more than one category)

import os

screens = {} # dictionary where the keys are screens we encountered so far

count = 0
for f in os.listdir("GP-Categories"):
    for line in open(os.path.join("GP-Categories", f)).read().splitlines():
        if line not in screens:
            count += 1
            screens[line] = [f]
        else:
            screens[line].append(f)

dup = 0
for key in screens:
    if len(screens[key]) > 1:
        dup += 1 #print(key + " : " + str(screens[key]))

print(count)
print(str(dup) + " duplicates")
