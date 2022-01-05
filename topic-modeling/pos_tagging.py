import os
import csv
import time
import operator
import json

os.environ["CLASSPATH"] = "stanford-pos"
os.environ["STANFORD_MODELS"] = "stanford-pos"
from nltk.tag import StanfordPOSTagger
st = StanfordPOSTagger("english-bidirectional-distsim.tagger") 


unique_csv = open("unique.csv")
unique_csv_lines = unique_csv.read().splitlines()
unique_csv.close()




# list of lists; each list within this is a list of worsd
# i.e. elements of this list are ['a','b','c'],['word1','word2','word3']...
# for blank low captions, no list is appended
all_tokenized_low_captions = []





# iterate through key csv row by row (HIT by HIT)
for i, row in enumerate(csv.reader(unique_csv_lines, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)):
    if i != 0: # if this row is not the header
        low_captions = [row[2], row[3], row[4], row[5]]
        
        # list that gets appended to all_tokenized_low_captions

        
        for cap in low_captions:
            if cap.strip() != "":
                tokens = cap.split()
                if len(tokens) > 0:
                    all_tokenized_low_captions.append(tokens)

# [('below', 'IN'), ('the', 'DT'), ('browse', 'JJ'), ('option', 'NN'), ('there', 'EX'), ('is', 'VBZ'), ('a', 'DT'), ('link', 'NN'), ('for', 'IN'), ('login', 'NN')]

tag_fname = "pos_tagged_low_captions.json"
if os.path.isfile(tag_fname):
    print("Found saved pos file `%s`! Loading..."%(tag_fname))
    tag_file = open(tag_fname,"r")
    all_pos_tagged_caps = json.loads(tag_file.read())
    tag_file.close()
    print("Loaded POS tags!")
else:
    print("Tagging all captions by POS...")
    start_time = time.time()
    all_pos_tagged_caps = st.tag_sents(all_tokenized_low_captions)
    end_time = time.time()
    print("Done POS tagging all captions  in %0.4f s!"%(end_time-start_time))

    tag_file = open(tag_fname,"w")
    tag_file.write(json.dumps(all_pos_tagged_caps))
    tag_file.close()
    
    print("Successfully wrote to `%s`!"%(tag_file.name))


# dictionary mapping a POS prefix of variable length to a count that we'll sort
# i.e. "IN/DT/NN/IN/DT/NN" -> 99
prefix_dict = {}

# dictionary mapping a POS suffix of variable length to a count that we'll sort
# i.e. "IN/DT/NN/IN/DT/NN" -> 99
suffix_dict = {}

# populate prefix_dict and suffix_dict

print("Counting common prefixes/suffixes...")

for tagged_cap in all_pos_tagged_caps:
    # i.e. tagged_cap = [('at', 'IN'), ('the', 'DT'), ('top', 'NN'), ...]
    
    for subseq_length in range(3, 15+1): # 3 to 15 inclusive is arbitrary
        # we can only take a subseq of a length <= to the length of tagged_cap
        # (where subseq_lengths correspond to words)
        
        if subseq_length <= len(tagged_cap):
            
            # get the POS prefix of this caption of length `subseq_length`; i.e. "IN/DT/NN"
            
            pos_pref = "/".join([tagged_cap[i][1] for i in range(0, subseq_length)])
            
            # get the POS suffix of this caption of length `subseq_length`; i.e. "IN/DT/NN"
            
            pos_suf = "/".join([tagged_cap[len(tagged_cap)-subseq_length+i][1] for i in range(0, subseq_length)])
            
            
            #print("subseq_length = %d"%(subseq_length))
            #print(tagged_cap)
            #print(pos_pref)
            #print(pos_suf)
            #input()

            # if the prefix is not in prefix_dict, add it
            if pos_pref not in prefix_dict:
                prefix_dict[pos_pref] = 0
            
            # if the suffix is not in suffix_dict, add it
            if pos_suf not in suffix_dict:
                suffix_dict[pos_suf] = 0
            
            # increment the counts on each
            prefix_dict[pos_pref] += 1
            suffix_dict[pos_suf] += 1
            
print("Done counting common prefixes/suffixes!")

# now create two sorted lists `prefix_list` and `suffix_list`
# that contain the top frequent prefixes/suffixes
# i.e. prefix_list = ["IN/DT/NN/IN/DT/NN", ...]

prefix_list = sorted(prefix_dict.items(), key=operator.itemgetter(1), reverse=True)
suffix_list = sorted(suffix_dict.items(), key=operator.itemgetter(1), reverse=True)

print(str(prefix_list)[0:2000])
print(str(suffix_list)[0:2000])


# now strip away the counts from prefix_list and suffix_list
# (i.e. [('NN/IN/NN', 896), ('VBZ/DT/NN/TO/VB/DT/NN', 762)] -> ['DT/NN/VBD/NN', 'DT/NN/MD/VB'])

# and only keep the top N prefixes/suffixes
TOP_N_SUBSEQ = 100 # we keep the TOP_N SUBSEQs in the prefix and suffix lists

prefix_list = [elem[0] for elem in prefix_list]
suffix_list = [elem[0] for elem in suffix_list]

print(str(prefix_list)[0:2000])
print(str(suffix_list)[0:2000])

print("Exiting...")
exit()

def list_length(a):
    return len(a)

# a blacklist of phrases using POS tagging
# which will be sorted by list lengths
# so that bigger prefixes get cut out before smaller ones do
blacklist = [["IN", "DT", "NN", "IN", "DT", "NN"], # in the bottom of the screen
             ["IN", "DT", "NN", "IN", "DT", "NN", "EX", "VBZ"], # in the bottom of the screen there is
             ["IN", "DT", "NN", "IN", "DT", "NN", "VBZ", "DT"], # at the bottom of the screen is a
            ]

# sort by list length reversed (longer lists come first in the blacklist)
blacklist.sort(key=list_length, reverse=True)



# list of lists; each list in this list has four low level captions whose blacklisted part of speech patterns have been taken away
# this will be used in writing out the unique_pos_blacklisted.csv
all_filtered_low_captions = []


'''

#('in', 'IN'), ('the', 'DT'), ('center', 'NN'), ('of', 'IN'), ('the', 'DT'), ('screen', 'NN')

# for each low caption
for j, cap in enumerate(low_captions):
    if cap != "":
        pos_tagged_cap = st.tag_sents([cap.split()])
        
        print(pos_tagged_cap)
        
        # for each blacklisted POS LIST
        for POS_LIST in blacklist:
            
            # make sure the length of the caption itself is at least as long as the POS LIST
            if (len(pos_tagged_cap) >= len(POS_LIST)):
                
                
                
                # if the POS_LIST is a prefix of this caption
                if [pos_tagged_cap[k][1] for k in range(len(POS_LIST))] == POS_LIST:
                    #print(pos_tagged_cap)
                    low_captions[j] = " ".join([pos_tagged_cap[k][0] for k in range(len(POS_LIST), len(pos_tagged_cap))])
                    #print(low_captions[j])
                    break; # stop going through the blacklist
                # if the POS_LIST is a suffix of this caption
                elif [pos_tagged_cap[len(pos_tagged_cap)-len(POS_LIST)+k][1] for k in range(len(POS_LIST))] == POS_LIST:
                    #print(pos_tagged_cap)
                    low_captions[j] = " ".join([pos_tagged_cap[k][0] for k in range(0, len(POS_LIST))])
                    #print(low_captions[j])
                    break; # stop going through the blacklist



    # append our filtered captions to the filtered captions list
    all_filtered_low_captions.append(low_captions)

'''


