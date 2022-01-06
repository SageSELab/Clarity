import csv
from re import search
import string

fields = ['Checkpoints', 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'CIDEr']
result_csv = "scores.csv"

with open(result_csv, 'a') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(fields)
    for x in range(999,2000,1000):
        filestr = "low-captions-check-" + str(x) + "-results.txt"
        filename = open(filestr, "r")
        lines = filename.readlines()
        row = [] 
        row.append(str(x))

        for line in lines:
            if search("^Bleu_1", line):
                row.append(line[line.find(" ")+1:line.find("\n")]);
            elif search("^Bleu_2", line):
                row.append(line[line.find(" ")+1:line.find("\n")]);
            elif search("^Bleu_3", line):
                row.append(line[line.find(" ")+1:line.find("\n")]);
            elif search("^Bleu_4", line):
                row.append(line[line.find(" ")+1:line.find("\n")]);
            elif search("^METEOR", line):
                row.append(line[line.find(" ")+1:line.find("\n")]);
            elif search("^CIDEr", line):
                row.append(line[line.find(" ")+1:line.find("\n")]);
        print(row) 
        
        writer.writerow(row)
        
