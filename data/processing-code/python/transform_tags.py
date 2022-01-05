#Script to create a mechanical turk csv from a manually tagged csv (does not overwrite the original manually tagged csv)

import os
import sys
import csv
import re

def print_usage():
    print("\nReads in a manually tagged csv and creates a mechanical turk csv.")
    print("\nusage: python " + __file__ + " <manually tagged csv>")
    print("\nex: python " + __file__ + " tags.csv \n")
    exit()

if len(os.sys.argv) < 2:
    print_usage()

fname = os.sys.argv[1]

if fname[len(fname)-4:len(fname)] != ".csv":
    print("\nerror: input file must be a csv!\n")
    exit()




std_header = ['HITId', 'HITTypeId', 'Title', 'Description', 'Keywords', 'Reward', 'CreationTime', 'MaxAssignments', 'RequesterAnnotation', 'AssignmentDurationInSeconds', 'AutoApprovalDelayInSeconds', 'Expiration', 'NumberOfSimilarHITs', 'LifetimeInSeconds', 'AssignmentId', 'WorkerId', 'AssignmentStatus', 'AcceptTime', 'SubmitTime', 'AutoApprovalTime', 'ApprovalTime', 'RejectionTime', 'RequesterFeedback', 'WorkTimeInSeconds', 'LifetimeApprovalRate', 'Last30DaysApprovalRate', 'Last7DaysApprovalRate', 'Input.image_url', 'Answer.HighLevel', 'Answer.LowLevel1', 'Answer.LowLevel2', 'Answer.LowLevel3', 'Answer.LowLevel4', 'Approve', 'Reject']

mantag_text = open(fname, "r").read()

mantag_text = mantag_text.replace("\;",":") #replace escaped semicolon with regular colon for the purpose of parsing
mantag_text = mantag_text.replace("\t"," ") #replace all tabs with spaces
mantag_lines = mantag_text.splitlines() #read in the manually tagged csv and split lines



mechtag_out = "" #string containing the mechanical turk output of the manually tagged data

for i in range(len(std_header)-1): #for each header, wrap it in the mechical turk format of "word", 
    mechtag_out += "\"" + std_header[i] + "\","


mechtag_out += "\"" + std_header[len(std_header)-1] + "\"\n" # do the last entry manually to prevent an extra comma

    
for row in csv.reader(mantag_lines, delimiter=';',  skipinitialspace=True): #go through csv row by row
    
    
    
    #descriptions = [row[28], row[29], row[30], row[31], row[32]] #descriptions is a row of the descriptions. In order: [HighLevel1, LowLevel1, LowLevel2, LowLevel3, LowLevel4]
    
    lowlevels = row[4].split(".")
    
    lowlevels_truncated = []
    
    
    
    for desc in lowlevels:
        if desc.strip() != '': #i.e. if the string is not only whitespace AND the string is not empty
            lowlevels_truncated.append(desc + ".") #append the string to the lowlevels_truncated list (adding a final period)
            
            if len(lowlevels_truncated) == 4: #break if we filled up lowlevels_truncated
                break;
    
    
    if len(lowlevels_truncated) < 4: #if we didn't have at least 4 low level descriptions,
        for i in range(4 - len(lowlevels_truncated)):
            lowlevels_truncated.append('') # just append empty strings until the lowlevel truncated size is 4
    
    assert(len(lowlevels_truncated) == 4) #sanity check; its length is always 4 anyway

   
    if len(row) == 5:
        row_out = "" #row to append to mechtag_out
        
        for i in range(32):
            if i == 16:
                row_out += "\"" + "Approved" + "\"," #append "Approved" (manually tagged = approved already)
            elif i == 27: #27 is the filename
                row_out += "\"" + row[0].strip() + "\"," #append filename
            elif i == 28: #28 is the high level description
                row_out += "\"" + row[3].strip() + "\"," #append high level description
            elif i == 29: #29 is the low level descriptions
                row_out += "\"" + lowlevels_truncated[0].strip() + "\"," #append lowlevel1
            elif i == 30: #29 is the low level descriptions
                row_out += "\"" + lowlevels_truncated[1].strip() + "\"," #append lowlevel2
            elif i == 31: #29 is the low level descriptions
                row_out += "\"" + lowlevels_truncated[2].strip() + "\"," #append lowlevel3
            else:
                row_out += "\"" + "NA" + "\"," #NA because this is an artificial mech turk file
        
        
        row_out += "\"" + lowlevels_truncated[3].strip() + "\"\n" #append last entry to row_out manually to prevent an extra comma
        
        
        mechtag_out += row_out; #append the row to mechtag_out


out_name = "batch_manually_tagged.csv"

mechtag_csv = open(out_name,"w")
mechtag_csv.write(mechtag_out)
mechtag_csv.close()

print("\nWrote mechanical turk csv to " + out_name + "\n")




