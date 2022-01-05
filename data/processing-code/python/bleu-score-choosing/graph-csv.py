# message that's printed when the script is run without arguments
info_msg = '''
Script to graph the Bleu-1, Bleu-2, Bleu-3, Bleu-4, and avg_score
of two given "summarized" bleu score csvs (such as `scores_bo2txtRC_test`)
where the format of a "summarized" csv is 
"preds_file_name, avg_bleu1, avg_bleu2, avg_bleu3, avg_bleu4, avg_score, Best?"

NOTE: The graph is generated using matplotlib and it will open automatically;
      the user may then save the graph as a png.
      
NOTE 2: For Clarity, we only generated these graphs on the test set.
'''

#@author Ali Yachnes

import sys
import os
import csv
from matplotlib import pyplot

# the two plots we're interested in are:
# scores-test/scores_hi2txtGP_test.csv scores-test/scores_no_finetune_high_test.csv
# scores-test/scores_lo2txtRC_test.csv scores-test/scores_finetune_low_test.csv

if len(sys.argv) != 5:
    print(info_msg)
    print("usage: python3 " + __file__ + " <scores1 csv> <scores2 csv>\n")
    print("i.e. python3 " + __file__ + " scores-test/scores_hi2txtGP_test.csv scores-test/scores_no_finetune_high_test.csv scores-test/scores_lo2txtRC_test.csv scores-test/scores_finetune_low_test.csv\n")
    exit()


input_csv_1_name = sys.argv[1]
input_csv_2_name = sys.argv[2]
input_csv_3_name = sys.argv[3]
input_csv_4_name = sys.argv[4]

for input_csv_name in [input_csv_1_name, input_csv_2_name, input_csv_3_name, input_csv_4_name]:
    if not os.path.isfile(input_csv_name):
        print("Error: `%s` is not a file."%input_csv_name)
        exit()
    elif input_csv_name[-4:] != ".csv":
        print("Error: `%s` does not end in .csv"%(input_csv_name))
        exit()

###### before getting the data from the csvs, set up some pyplot preliminaries ######

# Common sizes: (10, 7.5) and (12, 9)    
pyplot.figure(figsize=(12, 9))    

# change x and y ranges 
pyplot.xlim(1, 500000)
pyplot.ylim(0, .4)

# change x and y ticks
step_size = 1
pyplot.yticks([v/10.0 for v in range(0,4+1,step_size)], [str(x/10) for x in range(0, 4+1, step_size)], fontsize=13)    
pyplot.xticks(fontsize=14)


# get rid of the plot frame lines
ax = pyplot.subplot(111)    
ax.spines["top"].set_visible(False)    
#ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
#ax.spines["left"].set_visible(False)    

# Ensure that the axis ticks only show up on the bottom and left of the plot.    
# Ticks on the right and top of the plot are generally unnecessary chartjunk. 
ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left() 


############# now get data from the csvs #############

# extract string names from the inputted csvs; used later in titling the plot
plot1_name = input_csv_1_name.split("/")[-1][0:-4]
plot2_name = input_csv_2_name.split("/")[-1][0:-4]
plot3_name = input_csv_3_name.split("/")[-1][0:-4]
plot4_name = input_csv_4_name.split("/")[-1][0:-4]

# extract the X - Y data from each csv
# each of the Y lists contain 5 lists: bleu-1, bleu-2, bleu-3, bleu-4, avg_score
# while the X lists have the iteration number associated with each nested list in plot_?_Y
plot_1_X = []
plot_1_Y = [[],[],[],[],[]]

plot_2_X = []
plot_2_Y = [[],[],[],[],[]]

plot_3_X = []
plot_3_Y = [[],[],[],[],[]]

plot_4_X = []
plot_4_Y = [[],[],[],[],[]]

csv_1 = open(input_csv_1_name,"r")
csv_1_lines = csv_1.read().splitlines()
csv_1.close()

csv_2 = open(input_csv_2_name,"r")
csv_2_lines = csv_2.read().splitlines()
csv_2.close()

csv_3 = open(input_csv_3_name,"r")
csv_3_lines = csv_3.read().splitlines()
csv_3.close()

csv_4 = open(input_csv_4_name,"r")
csv_4_lines = csv_4.read().splitlines()
csv_4.close()


for csv_num in range(0,3+1):
    
    curr_plot_name = None
    curr_csv_lines = None
    curr_plotX = None
    curr_plotY = None
    curr_color = None
    curr_style = None
    
    # do csv1
    if csv_num == 0:
        curr_csv_lines = csv_1_lines
        curr_plotX = plot_1_X
        curr_plotY = plot_1_Y
        curr_plot_name = plot1_name
        curr_style = "-"
        curr_color = (255,127,14)
    # do csv2
    elif csv_num == 1:
        curr_csv_lines = csv_2_lines
        curr_plotX = plot_2_X
        curr_plotY = plot_2_Y 
        curr_plot_name = plot2_name
        curr_style = "--"
        curr_color = (44,160,44)
    # do csv3
    elif csv_num == 2:
        curr_csv_lines = csv_3_lines
        curr_plotX = plot_3_X
        curr_plotY = plot_3_Y 
        curr_plot_name = plot3_name
        curr_style = "-."
        curr_color = (214,39,40)
    # do csv4
    elif csv_num == 3:
        curr_csv_lines = csv_4_lines
        curr_plotX = plot_4_X
        curr_plotY = plot_4_Y 
        curr_plot_name = plot4_name
        curr_style = ":"
        curr_color = (11,79,190)
                
                
                
    # for each row in whichever csv we're currently dealing with
    for i, row in enumerate(csv.reader(curr_csv_lines, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)):
        if i != 0: # if this row is not the header
            # order of headers in the csv: preds_file_name, avg_bleu1, avg_bleu2, avg_bleu3, avg_bleu4,avg_score,Best?
            # we only need bleu-1,2,3,4, and avg_score
            # as well as the iteration number which is embedded in preds_file_name
            
            iteration_number = 0
            
            # if this is a ntk2 csv, then multiply the number in `preds_file_name` by 2000
            if row[0].find("finetune") != -1:
                iteration_number = int(row[0].split("_")[-2])*2000
            else:
                # if this is an im2txt csv, take the iteration number as it is
                iteration_number = int(row[0].split("_")[-2])
                
            
            # add the iteration number to the X axis
            curr_plotX.append(iteration_number)
            
            # add all of the bleu scores to the Y axes
            curr_plotY[0].append(float(row[1])) # Bleu-1
            curr_plotY[1].append(float(row[2])) # Bleu-2
            curr_plotY[2].append(float(row[3])) # Bleu-3
            curr_plotY[3].append(float(row[4])) # Bleu-4
            curr_plotY[4].append(float(row[5])) # avg_score
            
            
            
    labels = ["Bleu-1","Bleu-2","Bleu-3","Bleu-4","Average"]
    colors = [(253, 164, 20), (214, 39, 40), (255, 127, 14), (158, 218, 229), (44, 160, 44)]
    
    for c in range(len(colors)):
        colors[c] = (colors[c][0]/255.0,colors[c][1]/255.0,colors[c][2]/255.0)
        
    # the data from this csv has been extracted; now plot everything
    for j,y_values in enumerate(curr_plotY):
        
        semantic_title = ""
        
        is_im2txt = False
        
        if (curr_plot_name.find("finetune") != -1):
            if (curr_plot_name.find("low") != -1):
                semantic_title = "neuraltalk low"
            elif (curr_plot_name.find("high") != -1):
                semantic_title = "neuraltalk high"
        else:
            is_im2txt = True
            if (curr_plot_name.find("lo2txt") != -1):
                semantic_title = "im2txt low"
            elif (curr_plot_name.find("hi2txt") != -1):
                semantic_title = "im2txt high"
        
        
        line_style = ("--" if is_im2txt else "-")
        if j == 4:
            curr_color = (curr_color[0]/255.0, curr_color[1]/255.0, curr_color[2]/255.0)
            pyplot.plot(curr_plotX, y_values, label=(semantic_title+" "+labels[j]), linestyle=curr_style, color=curr_color, linewidth=2.7)


# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])


pyplot.title("Average BLEU 1, 2, 3, 4 for Neuraltalk2 and Im2txt")
pyplot.ylabel('Score')
pyplot.xlabel('Iteration')
ax.legend(loc='center left', bbox_to_anchor=(1.03, 0.5))

fig = pyplot.gcf()
fig.canvas.set_window_title("BLEU Score Graph")

# bbox_inches="tight" removes all the extra whitespace on the edges of your plot.    

# come up with a name for the png
png_name = plot1_name+"-"+plot2_name+"-"+plot3_name+"-"+plot4_name+".pdf"


pyplot.savefig(png_name, bbox_inches="tight")
pyplot.show()
print("\nSaved %s!\n"%(png_name))
