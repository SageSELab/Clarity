# script to make 3 box plots, each of which has 11 boxes;
# the boxes are in this order:

'''
# 0) im2txt low model predicted caption
# 1) im2txt high model predicted caption
# 2) im2txt combined model predicted caption
# 3) seq2seq low model predicted caption
# 4) seq2seq high model predicted caption
# 5) seq2seq combined model predicted caption
# 6) neuraltalk low model predicted caption
# 7) neuraltalk high model predicted caption
# 8) neuraltalk combined model predicted caption
# 9) groundtruth low caption (randomly sampled from one of four NONEMPTY captions)
# 10) groundtruth high caption (given that the high level caption is NONEMPTY)
'''

import matplotlib.pyplot as plt
import numpy as np
import json


data = []

# fetch our data from a json

underst_json = open("underst_likert_raw_data.json", "r")

# get json data as a dictionary
raw_data = json.load(underst_json)
underst_json.close()


# labels for each boxplot; ordered

box_labels = ["groundtruth low", "groundtruth high", "seq2seq combined",
                "seq2seq low", "seq2seq high", "neuraltalk combined",
                 "neuraltalk low", "neuraltalk high", "im2txt combined",
                 "im2txt low", "im2txt high"]

assert(len(box_labels) == 11)

# raw_data is a dictionary, and box_labels contains the keys to the dictionary
# (except the space should be replaced with an underscore)

for lbl in box_labels:
    key = lbl.replace(" ", "_")
    
    # make sure the key is actually present
    assert(key in raw_data)
    
    data.append(raw_data[key])


plt.figure()
plt.boxplot(data, vert=False, labels=box_labels, showmeans=True)
plt.xticks([1,2,3], ["Easy", "Somewhat", "Hard"])

ax = plt.subplot(111)
ax.set_axisbelow(True)
ax.set_title("Distributions of 'Readable' Likerts")
ax.set_xlabel("Readability")
ax.set_ylabel("Caption Type")





# now save and display

plt_name = "understandable_likert_boxplot.pdf"
plt.savefig(plt_name, bbox_inches="tight")
print("Saved boxplot as `%s`"%(plt_name))

print("Showing boxplot...")
plt.show()
print("Done.")
