# script to get the average ratings for each type of caption from the survey

import os
import csv
import json

if len(os.sys.argv) < 3:
    print("Script to get the average ratings for each type of caption from the survey.")
    print("Note: a key file that was generated by the survey is needed for this script")
    print("that maps each HIT to a mapping of the order of captions in the HIT.")
    print("\nusage: python3 " + __file__ + " <input batch csv> <survey key csv>")
    print("ex: python3 " + __file__ + " batches/batch_final.csv ../survey_key.csv")
    exit()



input_batch_csv_name = os.sys.argv[1]

if not os.path.isfile(input_batch_csv_name):
    print("No such file: %s"%(input_batch_csv_name))
    exit()


input_key_csv_name = os.sys.argv[2]

if not os.path.isfile(input_key_csv_name):
    print("No such file: %s"%(input_key_csv_name))
    exit()



batch_csv = open(input_batch_csv_name)
batch_csv_lines = batch_csv.read().splitlines()
batch_csv.close()


key_csv = open(input_key_csv_name)
key_csv_lines = key_csv.read().splitlines()
key_csv.close()


# mapping of semeru hit ID to the order of captions; all integers
# (i.e. semeru_id -> [a,b,c,d,e,f,g,h,i,j,k])
# (i.e. 1 -> [7, 2, 0, 8, 4, 9, 1, 6, 3, 5, 10])
# (this is built from the key file)
SemeruID_to_mapping = {}

# iterate through key csv row by row (HIT by HIT)
for i, row in enumerate(csv.reader(key_csv_lines, delimiter=',', skipinitialspace=True)):
    
    # skip the header row
    if (i != 0):
        
        semeru_id = int(row[0]);
        
        SemeruID_to_mapping[semeru_id] = [int(row[j]) for j in range(1, 11+1)]
        
        #print(SemeruID_to_mapping[semeru_id]);
            
# ordering of the likert columns in the csvs

'''
theses are the values we want to average:

"This description accuractley describes the functionality of the screenshot"
1 - 5 where 1 is "Strongly Disagree" and 5 is "Strongly Agree"
(columns 51 to 61 inclusive with 0 based indexing)
Answer.likert1	Answer.likert10	Answer.likert11	Answer.likert2	Answer.likert3	Answer.likert4	Answer.likert5	Answer.likert6	Answer.likert7	Answer.likert8	Answer.likert9

"Considering the content of the descritpion, do you think that the description: "
1 - 3 where 1 is "Is easy to read and understand", 2 is "Is somewhat readable and understandable", and 3 is "Is hard to read and understand"
(columns 95 to 105 inclusive with 0 based indexing)
Answer.understandability1	Answer.understandability10	Answer.understandability11	Answer.understandability2	Answer.understandability3	Answer.understandability4	Answer.understandability5	Answer.understandability6	Answer.understandability7	Answer.understandability8	Answer.understandability9

"Considering the content of the descritpion, do you think that the description:"
1 - 3 where 1 is "Has no unnessecary information", 2 is "Has some unnessecary information" and 3 is "Has a lot of unnessecary information"
(columns 106 to 116 inclusive with 0 based indexing)
Answer.unnesc_info1	Answer.unnesc_info10	Answer.unnesc_info11	Answer.unnesc_info2	Answer.unnesc_info3	Answer.unnesc_info4	Answer.unnesc_info5	Answer.unnesc_info6	Answer.unnesc_info7	Answer.unnesc_info8	Answer.unnesc_info9
'''

accurate_likert_cols = ["Answer.likert1", "Answer.likert10", "Answer.likert11", "Answer.likert2", "Answer.likert3", "Answer.likert4", "Answer.likert5", "Answer.likert6", "Answer.likert7", "Answer.likert8", "Answer.likert9"]
unnesc_likert_cols = ["Answer.unnesc_info1", "Answer.unnesc_info10", "Answer.unnesc_info11", "Answer.unnesc_info2", "Answer.unnesc_info3", "Answer.unnesc_info4", "Answer.unnesc_info5", "Answer.unnesc_info6", "Answer.unnesc_info7", "Answer.unnesc_info8", "Answer.unnesc_info9"]
readability_likert_cols = ["Answer.understandability1", "Answer.understandability10", "Answer.understandability11", "Answer.understandability2", "Answer.understandability3", "Answer.understandability4", "Answer.understandability5", "Answer.understandability6", "Answer.understandability7", "Answer.understandability8", "Answer.understandability9"]

# map each string from each set of likert columns to the number in the string minus 1
# such that a 0 corresponds to a likert scale from the 0th screenshot of this HIT
for i in range(len(accurate_likert_cols)):
    accurate_likert_cols[i] = int(accurate_likert_cols[i][len("Answer.likert"):])-1

for i in range(len(readability_likert_cols)):
    readability_likert_cols[i] = int(readability_likert_cols[i][len("Answer.understandability"):])-1

for i in range(len(unnesc_likert_cols)):
    unnesc_likert_cols[i] = int(unnesc_likert_cols[i][len("Answer.unnesc_info"):])-1

# each list after this process should be:
# [0, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8]
# (they all happened to have the same ordering in the mech turk csv; alphabetical)


# likert orderings; i.e. if an accurate_likert in the csv has a score of n, this corresponds to the 
# (n-1)th element of the `accurate_likert` list; same goes for the other two
accurate_likert_resps = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
readability_likert_resps = ["Is easy to read and understand", "Is somewhat readable and understandable", "Is hard to read and understand"]
unnesc_likert_resps = ["Has no unnessecary information", "Has some unnessecary information", "Has a lot of unnessecary information"]

'''
Note: each integer corresponds to:
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

# mapping from the integer to the caption type; used when we need
# to write out strings with the semantic representation of a caption index
cap_num_to_label = {
                         0:  "im2txt_low",
                         1:  "im2txt_high",
                         2:  "im2txt_combined",
                         3:  "seq2seq_low",
                         4:  "seq2seq_high",
                         5:  "seq2seq_combined",
                         6:  "neuraltalk_low",
                         7:  "neuraltalk_high",
                         8:  "neuraltalk_combined",
                         9:  "groundtruth_low",
                         10: "groundtruth_high"
                    }

# a list where the ith element corresponds to the list in the comment above,
# and where each element is a dictionary like:
'''
{
    "accurate": 4.4
    "unnesc": 2.7
    "readability": 1.4
}
'''
avg_likert_scores = []

# each of the following three lists will contain a count of the ratings for each caption
# i.e. each list is of size 11 (0 - 10 inclusive)
# in the first list, ratings range from 1 - 5 so each element of the list is a list of 5 elements
# in the other two lists, ratings range from 1 - 3 so each element of each list is a list of 3 elements

perc_acc_scores = []
perc_rdb_scores = []
perc_unn_scores = []

# initialize each type of caption to all 0s
for i in range(0, 10+1):
    
    new_dict = {}
    new_dict["accurate"] = 0.0
    new_dict["unnesc"] = 0.0
    new_dict["readability"] = 0.0
    
    avg_likert_scores.append(new_dict)
    
    perc_acc_scores.append([0.0, 0.0, 0.0, 0.0, 0.0])
    perc_rdb_scores.append([0.0, 0.0, 0.0])
    perc_unn_scores.append([0.0, 0.0, 0.0])



# order of columns of freeform responses, 0 indexed; 
# this is the same for all freeform responses since mech turk 
# uses alphabetical ordering
ff_cols = [0, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8]



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

# a list of 11 elements, each a list of rows (each a csv)
# where we use the above mapping to tell what csv each element is
ff_resp_csvs = []

for i in range(11):
    # append the header row onto each csv
    ff_resp_csvs.append([["caption_type", "What aspects of this description are accurate?", "What aspects of this description are NOT accurate?", "How could this description be improved?"]])



# the number of approved HITs; this is used in calculating the average score
# since we need to divide by the number of ratings to get the average
NUM_APPROVED_HITS = 0
ids = {}
# iterate through batch csv row by row (HIT by HIT)
for i, row in enumerate(csv.reader(batch_csv_lines, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)):
    
    # skip the header row
    if (i != 0):
        status = row[16].lower().strip(); # "submitted", "rejected", or "approved"
        
        # we only want to analyze the approved HITs
        if status == "approved":
            NUM_APPROVED_HITS += 1
            
            # get the semeru_id of this HIT
            
            semeru_id = int(row[27])
            ids[semeru_id] = "a"
            # get the order of captions for this HIT
            order_mapping = SemeruID_to_mapping[semeru_id]
            

            
            # accurate likert is 51 to 61 w/ 0-based indexing
            # readability likert is 95 to 105 w/ 0 based indexing
            # unnesc info likert is 106 to 116 w/ 0 based indexing
            # and all of the likert col lists are [0, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8]
            
            

            
            # sum up all the accurate likerts
            for col_num in range(51,61 + 1):
                screen_num = accurate_likert_cols[col_num-51] # screen num is 0 to 10 inclusive
                
                # get which caption this actually is
                # i.e. if cap_num is 0 then we know that it's `im2txt low model predicted caption`
                # (see the above mapping)
                cap_num = order_mapping[screen_num]
                rating = int(row[col_num])
                avg_likert_scores[cap_num]["accurate"] += rating
                
                # add one to the corresponding acc scores rating column
                perc_acc_scores[cap_num][rating-1] += 1.0
            
            
            # sum up all the unnesc info likerts
            for col_num in range(106,116 + 1):
                screen_num = unnesc_likert_cols[col_num-106] # screen num is 0 to 10 inclusive
                
                # get which caption this actually is (0 to 10 inclusive)
                # i.e. if cap_num is 0 then we know that it's `im2txt low model predicted caption`
                # (see the above mapping)
                cap_num = order_mapping[screen_num]
                rating = int(row[col_num])
                avg_likert_scores[cap_num]["readability"] += rating
                
                perc_rdb_scores[cap_num][rating-1] += 1.0
                
                
            # sum up all the understandibility likerts
            
            for col_num in range(95,105 + 1):
                screen_num = readability_likert_cols[col_num-95] # screen num is 0 to 10 inclusive
            
                # get which caption this actually is (0 to 10 inclusive)
                # i.e. if cap_num is 0 then we know that it's `im2txt low model predicted caption`
                # (see the above mapping)
                cap_num = order_mapping[screen_num]
                rating = int(row[col_num])
                avg_likert_scores[cap_num]["unnesc"] += rating
                perc_unn_scores[cap_num][rating-1] += 1.0




            # we need to aggregate all the free form responses 
            # for each caption type (one caption type per csv)
            
            # What aspects of this description are accurate? / Answer.open_response_accurateX
            # range 62 to 72 inclusive
            
            # How could this description be improved? / Answer.open_response_improvementsX
            # range 73 to 83 inclusive
            
            # What aspects of this description are NOT accurate? / Answer.open_response_not_accurateX
            # range 84 to 94 inclusive
            
            for col_num in range(0, 10+1):
                screen_num = ff_cols[col_num] # screen num is 0 to 10 inclusive
            
                # get which caption this actually is (0 to 10 inclusive)
                # i.e. if cap_num is 0 then we know that it's `im2txt low model predicted caption`
                # (see the above mapping)
                
                cap_num = order_mapping[screen_num]
                
                acc_response = row[col_num+62]
                improv_response = row[col_num+73]
                not_acc_response = row[col_num+84]

                ff_resp_csvs[cap_num].append([cap_num_to_label[cap_num], acc_response, not_acc_response, improv_response])


 

# we expect this many HITs because the one survey we ran included 220 HITs
# if we were to run another survey in the future, we would need to change this value
if (NUM_APPROVED_HITS != 220):
    print("Error: expected 220 approved HITs. Exiting...")
    exit()




#print(("="*40) + " RESULTS " + ("=" * 40) + ("\n"))


'''
before calculating the averages, generate the actual raw data
for each of the likert scores
since we want to graph them as box plots
(i.e. we have the counts of each type of likert score, so we can
generate the data here; if "Strongly Agree" == 50.0, then we can
generate 50 data points with 5's for the `accuracy` likert)
'''

'''
    Note: the mapping for the "accurate" likert is:

    1 - Strongly Disagree (SD)
    2 - Disagree (D)
    3 - Neutral (N)
    4 - Agree (A)
    5 - Strongly Agree (SA)

'''

# generate raw data for "accurate" likert
# and save it as a json in the `likert` folder

# a dictionary mapping each key to its raw data
# i.e. keys are:
'''
0:  "im2txt_low",
1:  "im2txt_high",
2:  "im2txt_combined",
3:  "seq2seq_low",
4:  "seq2seq_high",
5:  "seq2seq_combined",
6:  "neuraltalk_low",
7:  "neuraltalk_high",
8:  "neuraltalk_combined",
9:  "groundtruth_low",
10: "groundtruth_high"
'''

# all 3 raw data distributions
raw_accurate_data = {}
raw_unnesc_data = {}
raw_underst_data = {}


# for each caption type
for i in range(len(avg_likert_scores)):

    # add an entry in raw_accurate_data for this caption type
    # this entry will hold the distribution of data points
    
    # i.e. key could be "groundtruth_low"
    key = cap_num_to_label[i]
    
    # initialize an empty list
    raw_accurate_data[key] = []
    raw_unnesc_data[key] = []
    raw_underst_data[key] = []
    
    for likert_num in range(len(perc_acc_scores[0])):
        # number of data points of this type
        num_acc_pts = int(perc_acc_scores[i][likert_num])
        
        # `num_acc_pts` times, add the current likert score
        # likert scores are 1 - 5 inclusive (not 0 to 4 inclusive)
        # so we must add 1
        for _ in range(num_acc_pts):
            raw_accurate_data[key].append(likert_num+1)

    for likert_num in range(len(perc_unn_scores[0])):
        # number of data points of this type
        num_unn_pts = int(perc_unn_scores[i][likert_num])
        
        # `num_unn_pts` times, add the current likert score
        # likert scores are 1 - 5 inclusive (not 0 to 4 inclusive)
        # so we must add 1
        for _ in range(num_unn_pts):
            raw_unnesc_data[key].append(likert_num+1)


    for likert_num in range(len(perc_rdb_scores[0])):
        # number of data points of this type
        num_rdb_pts = int(perc_rdb_scores[i][likert_num])
        
        # `num_rdb_pts` times, add the current likert score
        # likert scores are 1 - 5 inclusive (not 0 to 4 inclusive)
        # so we must add 1
        for _ in range(num_rdb_pts):
            raw_underst_data[key].append(likert_num+1)


# make sure all of our distributions have the same number of points (sanity check)
first_len = len(raw_accurate_data["im2txt_low"])

for key in raw_accurate_data:
    assert(len(raw_accurate_data[key]) == first_len)
    #print(key, raw_accurate_data[key])

accurate_json = open("likert/accurate_likert_raw_data.json", "w")
json.dump(raw_accurate_data, accurate_json) # write the json into a file
accurate_json.close()
print("Successfully wrote to `%s`!"%(accurate_json))


# make sure all of our distributions have the same number of points (sanity check)
first_len = len(raw_unnesc_data["im2txt_low"])

for key in raw_unnesc_data:
    assert(len(raw_unnesc_data[key]) == first_len)
    #print(key, raw_unnesc_data[key])

unn_json = open("likert/unnesc_likert_raw_data.json", "w")
json.dump(raw_unnesc_data, unn_json) # write the json into a file
unn_json.close()
print("Successfully wrote to `%s`!"%(unn_json))




# make sure all of our distributions have the same number of points (sanity check)
first_len = len(raw_underst_data["im2txt_low"])

for key in raw_underst_data:
    assert(len(raw_underst_data[key]) == first_len)
    #print(key, raw_underst_data[key])

und_json = open("likert/underst_likert_raw_data.json", "w")
json.dump(raw_underst_data, und_json) # write the json into a file
und_json.close()
print("Successfully wrote to `%s`!"%(und_json))








# now actually average out each score in avg_likert_scores (divide by NUM_APPROVED_HITS)
for i in range(len(avg_likert_scores)):
    avg_likert_scores[i]["accurate"] /= float(NUM_APPROVED_HITS)
    avg_likert_scores[i]["unnesc"] /= float(NUM_APPROVED_HITS)
    avg_likert_scores[i]["readability"] /= float(NUM_APPROVED_HITS)
    
    # make perc_acc_scores into percentages
    for j in range(len(perc_acc_scores[0])):
        perc_acc_scores[i][j] /= float(NUM_APPROVED_HITS)
        perc_acc_scores[i][j] *= 100
    
    # make perc_rdb_scores into percentages
    for j in range(len(perc_rdb_scores[0])):
        perc_rdb_scores[i][j] /= float(NUM_APPROVED_HITS)
        perc_rdb_scores[i][j] *= 100
    
    # make perc_unn_scores into percentages
    for j in range(len(perc_unn_scores[0])):
        perc_unn_scores[i][j] /= float(NUM_APPROVED_HITS)
        perc_unn_scores[i][j] *= 100
    
    #print("%s: avg_accuracy = %f, avg_unnesc = %f, avg_understandibility = %f"%(cap_num_to_label[i].replace(" ", "_"),avg_likert_scores[i]["accurate"],  avg_likert_scores[i]["unnesc"], avg_likert_scores[i]["readability"]))


#print("\nwhere accuracy of 1 is 'Strongly Disagree' and 5 is 'Strongly Agree'")
#print("and unnsec of 1 is 'Has no unnessecary information' and 3 is 'Has a lot of unnessecary information'")
#print("and understandibility of 1 is 'Easy to read and understand' and 3 is 'Hard to read and understand'.")

'''
accurate_likert_resps = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
readability_likert_resps = ["Is easy to read and understand", "Is somewhat readable and understandable", "Is hard to read and understand"]
unnesc_likert_resps = ["Has no unnessecary information", "Has some unnessecary information", "Has a lot of unnessecary information"]
'''

# now make a csv with just the average scores
# and a 3 csvs with the percentage of responses (one for accurate, one for unnesc, and one for readability)

avg_csv_header = ["caption_type", "average_accuracy", "average_unnecessary_info","average_understandability"]
csv_file = open("likert/likert_avg.csv", "w")
csv_writer = csv.writer(csv_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)
csv_writer.writerow(avg_csv_header)

for i in range(len(cap_num_to_label)):
    csv_writer.writerow([ cap_num_to_label[i], "%.2f"%avg_likert_scores[i]["accurate"],  "%.2f"%avg_likert_scores[i]["unnesc"], "%.2f"%avg_likert_scores[i]["readability"] ])
    

csv_writer.writerow(["where accuracy of 1 is 'Strongly Disagree' and 5 is 'Strongly Agree'", "and unnsec of 1 is 'Has no unnessecary information' and 3 is 'Has a lot of unnessecary information'", "and understandibility of 1 is 'Easy to read and understand' and 3 is 'Hard to read and understand'."])

csv_file.close()

print("Successfully wrote to %s."%(csv_file.name))





# accuracy
perc_acc_header = ["caption_type", "Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
csv_file = open("likert/likert_perc_acc.csv", "w")
csv_writer = csv.writer(csv_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)
csv_writer.writerow(perc_acc_header)

for i in range(len(cap_num_to_label)):
    csv_writer.writerow([ cap_num_to_label[i], "%.2f"%perc_acc_scores[i][0], "%.2f"%perc_acc_scores[i][1], "%.2f"%perc_acc_scores[i][2], "%.2f"%perc_acc_scores[i][3], "%.2f"%perc_acc_scores[i][4]])
    
csv_file.close()

print("Successfully wrote to %s."%(csv_file.name))





# readability
perc_rdb_header = ["caption_type", "Is easy to read and understand", "Is somewhat readable and understandable", "Is hard to read and understand"]

csv_file = open("likert/likert_perc_underst.csv", "w")
csv_writer = csv.writer(csv_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)
csv_writer.writerow(perc_rdb_header)

for i in range(len(cap_num_to_label)):
    csv_writer.writerow([ cap_num_to_label[i], "%.2f"%perc_rdb_scores[i][0], "%.2f"%perc_rdb_scores[i][1], "%.2f"%perc_rdb_scores[i][2]])
    
csv_file.close()

print("Successfully wrote to %s."%(csv_file.name))



# unnecessary information
perc_unn_header = ["caption_type", "Has no unnessecary information", "Has some unnessecary information", "Has a lot of unnessecary information"]

csv_file = open("likert/likert_perc_unnesc.csv", "w")
csv_writer = csv.writer(csv_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)
csv_writer.writerow(perc_unn_header)

for i in range(len(cap_num_to_label)):
    csv_writer.writerow([ cap_num_to_label[i], "%.2f"%perc_unn_scores[i][0], "%.2f"%perc_unn_scores[i][1], "%.2f"%perc_unn_scores[i][2]])
    
csv_file.close()

print("Successfully wrote to %s."%(csv_file.name))



# now write to free response csvs

for i, fr_csv in enumerate(ff_resp_csvs):
    out_fr_csv = open("free_resp/%s_free_resp.csv"%(cap_num_to_label[i]), "w")

    csv_writer = csv.writer(out_fr_csv, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)
    
    for row in fr_csv:
        csv_writer.writerow(row)
    
    out_fr_csv.close()
    
    
    print("Successfully wrote to `%s` !"%(out_fr_csv.name))
    
    


print("\nDone!")
