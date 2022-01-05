import sys
import os
import csv

# returns whether the given row is a header in the format we're looking for
def is_header(row):
    header = ['preds_file_name', ' avg_bleu1', ' avg_bleu2', ' avg_bleu3', ' avg_bleu4']

    if len(row) != len(header):
        return False
    
    # skip the first element since lo2txtRC has "score_file_name" instead of "preds_file_name" (insignificant)
    for i in range(1,len(row)):
        if row[i] != header[i]:
            print(item1,item2)
            return False
    
    
    
    return True

# used in sorting tuples by checkpoint number
# checkpoint number is assumed to be element 2 of the tuple
def sort_by_ckpt_num(tup):
    return tup[2]

# used in sorting tuples by average score
# average score is assumed to be element 1 of the tuple
def sort_by_avg_score(tup):
    return tup[1]
    
    
if len(sys.argv) < 2:
    print("Script to choose the best row in a directory of csv files containing BLEU scores.")
    print("The script indicates the best row by adding a `BEST` marker to it;\n")
    print("The best row is chosen based on the average of its bleu-1,2,3,4 scores.")
    print("WARNING: each csv in the directory will be overwritten when marking the best row.")
    print("\nusage: python3 " + __file__ + " <directory containing im2txt/ntk2 score csvs>")
    print("i.e. python3 " + __file__ + " ./scores\n")
    exit()


input_dir = sys.argv[1]

if not os.path.isdir(input_dir):
    print("Error: `%s` is not a directory."%input_dir)
    exit()

for fname in os.listdir(input_dir):
    if fname[-4:] == ".csv":
        # open the file as a csv
        
        csv_file = open(os.path.join(input_dir,fname), "r")
        reader = csv.reader(csv_file, delimiter=',')
        # close the file for reading
        
        
        csv_data = [row for row in reader]
        
        csv_file.close()
        
        assert (is_header(csv_data[0])), "Error: wrong header; wrong csv type."
        
        # now that we know the csv is correct, find the row with the best average score
        
        
        # add a header column for avg_score which is the average score for each row
        csv_data[0].append("avg_score")
        
        # add a "Best" header column that indicates whether the row is the best
        csv_data[0].append("Best?")
        
        avg_scores = []
        
        # skipping the header row, for each row
        for i in range(1,len(csv_data)):
            
            # get the ckpt number (works for both im2txt and ntk2)
            
            # key to search on when finding the checkpoint number
            key = ""
            
            if csv_data[i][0].find("_test_") != -1:
                key = "_test_"
            elif csv_data[i][0].find("_val_") != -1:
                key = "_val_"
            else:
                print("Error: found row with a `preds_file_name` that does not have `test` or `val` in the title.")
                exit()
            
            # index to start at
            start = csv_data[i][0].find(key)+len(key)
            
            # preds name from start to the end; i.e. its first character starts the ckpt number
            altered_preds_name = csv_data[i][0][start:]
            
            ckpt_num = int( altered_preds_name[0:altered_preds_name.find("_")] )
            
            #print(ckpt_num)
            
            avg_score = 0.0
            
            # skipping the first column (filename), for all 4 bleu scores
            for j in range(1,5): # don't include 5; col 1 to 4 inclusive
                avg_score += float(csv_data[i][j])
                
            # get the average
            avg_score /= 4
            
            # add the average score to this row
            csv_data[i].append(avg_score)
            # append a tuple with the row, avg_score, and the ckpt number
            avg_scores.append((i, avg_score, ckpt_num))
            
        # since python sort is stable, we can sort avg_scores twice
        
        # first sort by checkpoint ascending (lowest checkpoint first)
        # then sort by score descending (highest score first)
        # in this way, we get a list of the best scores that come from the lowest checkpoints,
        # since we don't want to have checkpoint 500k when 200k - 500k all have exactly the same score
        
        avg_scores.sort(key=sort_by_ckpt_num)
        avg_scores.sort(key=sort_by_avg_score, reverse=True)
        
        # at this point, avg_scores[0] is the best entry
        
        best_entry = avg_scores[0]
        
        best_row_num = best_entry[0]
        
        
        print("%s: %s, %s, %s, %s (avg=%f)"%(fname,csv_data[best_row_num][1],csv_data[best_row_num][2],csv_data[best_row_num][3],csv_data[best_row_num][4],best_entry[1]))
        
        # open the csv file for writing (time to overwrite it)
        
        csv_file = open(os.path.join(input_dir,fname), "w")
        
        csv_writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        
        
        for i in range(len(csv_data)):
            
            row = csv_data[i]
            
            # don't do this for the header row
            if i != 0:
                if i == best_row_num:
                    # true indicates that the model is the best
                    row.append("TRUE")
                else:
                    row.append("false")
                    
            csv_writer.writerow(row)
        
        csv_file.close()
        
        #print("Overwrote %s."%csv_file.name)
