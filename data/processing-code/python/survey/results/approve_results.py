'''

Simple script to read in a survey results csv downloaded from MTurk and
sort it by user.

Then, one user at a time the freeform responses are shown in the console
and stdinput is read to determine whether to approve or deny the submission.

Only one HIT per unique user is shown, and this HIT is chosen to be the
HIT with the longest freeform responses (hopefully the best). If this HIT
is rejected for some reason, the HIT with the next longest freeform responses
will be shown, and this continues until a HIT by the worker is approved.
Once a HIT by a duplicate worker is approved, all of the rest of the HITs
by this worker will be denied with the reason:
`DUPLICATE_WORKER_REJECT_REASON`

Ali Yachnes
ayachnes@email.wm.edu

Some common reasons to reject something:

Worker described ways to change the screenshot in the 'how could this description be improved' section. We want ways to improve the description of the screenshot.
Worker's responses for the 'How could this description be improved?' section were lacking in detail, sometimes describing ways to change the screenshot itself. The question wanted ways to change the written description of the screenshot.
Low quality, nonspecific responses to free form fields. Please give more detail in your responses.
Free form responses describe the app screenshots instead of the descriptions of the app screenshots. Please reread the instructions before attempting this HIT again.
Worker did not give specific ways to change the descriptions in the 'how could this description be improved' section.

'''



import os
import csv

# function to allow sorting by the total number of characters in response list
def sort_by_responses_length(elem):
    
    num_chars = 0
    
    # for each form response
    for response in elem["forms"]:
        num_chars += len(response)
    
    return num_chars
    
if len(os.sys.argv) < 2:
    print("Script to provide an interactive interface (using stdin) to go ", end="")
    print("through a downloaded survey batch csv and approve/reject submissions.")
    
    print("\nNOTE: this script will automatically skip over duplicate submissions ", end = "")
    print("and reject them with the reason 'This HIT may only be completed once per user.'")
    print("For the case where a single worker completes more than one HIT, the HIT with the ", end="")
    print("longest freeform responses will be the one to be shown to approve/reject.")
    print("If this one is rejected, the one with the next longest results will be shown, etc. ", end = "")
    print("until a HIT by the worker is finally approved, at which point all of the duplicate")
    print("submissions by the worker will be automatically rejected.")
    
    print("\nusage: python3 " + __file__ + " <input batch csv>")
    print("\nex: python3 " + __file__ + " batch_01.csv")
    exit()


input_batch_csv_name = os.sys.argv[1]

if not os.path.isfile(input_batch_csv_name):
    print("No such file: %s"%(input_batch_csv_name))
    exit()

batch_csv = open(input_batch_csv_name)
csv_lines = batch_csv.read().splitlines()
batch_csv.close()

# string containing the verbose description of why workers completing a HIT
# more than once would be rejected
DUPLICATE_WORKER_REJECT_REASON = "It was clearly stated that this HIT may only be completed once per worker, and that HITs after a worker's first would be rejected."

# dictionary mapping worker id to a list of dictionaries 

# i.e. "A1ZLMSH2TS8HWH" -> [ {"forms" : ["free1","free2","free3"], "row" : 6}, 
#                            {"forms" : ["free1","free2","free3"], "row" : 5}, 
#                            {"forms" : ["free1","free2","free3"], "row" : 3} ]

forms_by_workerid = {}

# dictionary with keys of workers who have had a submission approved before
# this will be used to automatically reject workers when processing the
# submitted responses, as only one submission per worker is allowed
old_workers = {}

# list providing the order of freeform responses;
# i.e. the 0th element in this list is a key describing the 0th element
# in any freeform response list in `forms_by_workerid`.

free_form_order = ['Answer.open_response_accurate1', 'Answer.open_response_accurate10', 
                   'Answer.open_response_accurate11', 'Answer.open_response_accurate2', 
                   'Answer.open_response_accurate3', 'Answer.open_response_accurate4', 
                   'Answer.open_response_accurate5', 'Answer.open_response_accurate6', 
                   'Answer.open_response_accurate7', 'Answer.open_response_accurate8', 
                   'Answer.open_response_accurate9', 'Answer.open_response_improvements1', 
                   'Answer.open_response_improvements10', 'Answer.open_response_improvements11', 
                   'Answer.open_response_improvements2', 'Answer.open_response_improvements3', 
                   'Answer.open_response_improvements4', 'Answer.open_response_improvements5', 
                   'Answer.open_response_improvements6', 'Answer.open_response_improvements7', 
                   'Answer.open_response_improvements8', 'Answer.open_response_improvements9', 
                   'Answer.open_response_not_accurate1', 'Answer.open_response_not_accurate10', 
                   'Answer.open_response_not_accurate11', 'Answer.open_response_not_accurate2', 
                   'Answer.open_response_not_accurate3', 'Answer.open_response_not_accurate4', 
                   'Answer.open_response_not_accurate5', 'Answer.open_response_not_accurate6', 
                   'Answer.open_response_not_accurate7', 'Answer.open_response_not_accurate8', 
                   'Answer.open_response_not_accurate9'];

assert(len(free_form_order) == 33)


# essentially a duplicate of the batch_csv, useful for writing out
# each element of this list is a row, and rows will be modified
# (approved/rejected) and then written to disk
out_csv_rows = []

# first populate all the hits by worker
for i, row in enumerate(csv.reader(csv_lines, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)):
    
    status = row[16].lower().strip(); # "submitted", "rejected", or "approved"
    worker_id = row[15]
    
    # if it's the header row or it's a "Submitted" HIT, we want to write
    # it out later, so add it to out_csv_rows now
    if (i == 0) or (status == "submitted"):
        out_csv_rows.append(row)
    
    if i != 0: # exclude the header
        
        # if this row is in the "Submitted" state, we want to process it
        
        if status == "submitted":

            # if this is the first time encountering the worker, 
            if not worker_id in forms_by_workerid:
                forms_by_workerid[worker_id] = []
                
            # list of freeform responses
            ff_responses = [row[i] for i in range(62, 94+1)]
            
            # we must have 33 freeform responses total
            assert(len(ff_responses) == len(free_form_order))
            
            
            # append a dictionary for this worker (each HIT gets its own dictionary)
            
            forms_by_workerid[worker_id].append({"forms":ff_responses,"row":len(out_csv_rows)-1})
        elif status == "approved":
            # if the status is approved, then add the worker to the 
            # old_workers dictionary to make sure we automatically
            # reject any responses by them in the 'submitted' status
            
            old_workers[worker_id] = True
    

# Now, for each worker, sort their list of lists of responses by the
# total characters in the list of responses; i.e. ["a","b","c"] would
# be a list of responses with length 3.

# This allows us to look at the "highest quality" lists of responses for
# a given worker first, increasing the likelihood that we won't have to
# look through many of their duplicate submissions (but ideally one worker
# will only have a single list of responses)

# assert that we have at least one HIT in the submitted state
assert(len(forms_by_workerid) != 0)

TOTAL_HITS = 0

# count the number of total HITS
for worker_id in forms_by_workerid:
    TOTAL_HITS += len(forms_by_workerid[worker_id])

# number of the current HIT
HIT_NUMBER = 0


print("")
                #forms_by_workerid
for worker_id in forms_by_workerid:
    
    forms_dict = forms_by_workerid[worker_id]
    
    # note: forms_dict is list of dictionaries: [ {"forms" : ["free1","free2","free3"], "row" : 6}, 
    #                                             {"forms" : ["free1","free2","free3"], "row" : 5}, 
    #                                             {"forms" : ["free1","free2","free3"], "row" : 3} ]
    
    forms_dict.sort(key=sort_by_responses_length)
    
    
    # whether to continue with the rest of the HITs by a worker
    # this should only be true after analyzing the first 33 freeforms
    # of a HIT if a) the worker did more than one HIT and b) the first 
    # HIT was rejected so there is still the possibility of approving
    # one of the duplicate hits
    
    continue_worker = True
    
    # for each HIT done by this unique worker
    
    for i in range(len(forms_dict)):
        
        HIT_NUMBER += 1
        
        # row (list) of the current HIT in `out_csv_rows`
        curr_HIT_row = out_csv_rows[forms_dict[i]["row"]]

        # if we already accepted a HIT from this user, 
        # reject the rest of their submissions automatically (duplicate)
        if (continue_worker == False) or (worker_id in old_workers):
            # append nothing in the approve column
            curr_HIT_row.append("")
            # append the reason in the reject column
            curr_HIT_row.append(DUPLICATE_WORKER_REJECT_REASON)
            
            print("!"*37)
            print("> Automatically rejected duplicate! <")
            print("!"*37)
            print("")
            
            # continue with the loop
            continue
            
        forms = forms_dict[i]["forms"]
        
        print("Worker ID: %s, User HIT %d/%d, Total HIT %d/%d"%(worker_id,(i+1),len(forms_dict),HIT_NUMBER,TOTAL_HITS))
        # format the responses in a nice way
        
        print(('=' * 50) + " What aspects of this description are accurate? " + ('=' * 50))
        print("")
        # print the "accurate" free responses
        for j in range(0,10+1):
            print("'"+forms[j]+"'")
        print("")
        
        print(('=' * 50) + " What aspects of this description are NOT accurate? " + ('=' * 50))
        print("")
        
        # print the "NOT accurate" free responses
        for j in range(22,32+1):
            print("'"+forms[j]+"'")
        
        print("")
        print(('=' * 50) + " How could this description be improved? " + ('=' * 50))
        print("")
        # print the "could be improved" free responses
        for j in range(11,21+1):
            print("'"+forms[j]+"'")
        
        print("")
        
        print(('=' * 148))
        
        approval = ""
        
        while (approval.lower() not in ["a","r"]):
            approval = input("(A)pprove/(R)eject? ").lower()
            
        # approve the HIT; go into the list of all the rows and append an "x"
        if approval == "a":
            # append an X in the approve column
            curr_HIT_row.append("x")
            # append nothing in the reject column
            curr_HIT_row.append("")
            
            # whether the worker has HITs left or not, we should 
            # not continue with them
            continue_worker = False
            
            print("Approved!")
            
        elif approval == "r":
            
            reject_reason = ""
            
            # while a valid rejection reason hasn't been given,
            while (len(reject_reason) <= 3):
                reject_reason = input("Rejection reason: ")
        
            # append nothing in the approve column
            curr_HIT_row.append("")
            # append the reason in the reject column
            curr_HIT_row.append(reject_reason)
            
            print("Rejected!")
        
        print("\n")



# Finally, go through each row in `out_csv_rows` and write it to a file
out_csv_name = ("approved_"+input_batch_csv_name)

if os.path.isfile(out_csv_name):
    
    overwrite = ""
    
    while overwrite.lower() not in ["y","n"]:
        overwrite = input("Warning: %s already exists. Are you sure you would like to overwrite it (y/N)? "%(out_csv_name))
        overwrite = overwrite.lower()
    if overwrite == "n":
        print("%s not written to, exiting."%(out_csv_name))
        exit()

# at this point we are sure we want to overwrite the csv

out_file = open(out_csv_name, "w")

csv_writer = csv.writer(out_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)

for row in out_csv_rows:
    csv_writer.writerow(row)
    
print("Successfully wrote to %s."%(out_csv_name))
print("Done!")
out_file.close()

    
