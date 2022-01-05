from matplotlib import pyplot

import re
import sys
import os

axes = {}

iterations = []
validation_loss = []
total_loss = []

if len(sys.argv) <= 1:
    print("usage: " + os.path.basename(__file__) + " <file> ")
    quit()

filename = sys.argv[1]



logfile = open(filename,"r").read()

labels = ['Iteration ', 'validation loss: ', 'total_loss: ']

for term in labels:
    pattern = re.compile(term + r'\d+.\d*')
    axes[term] = []
    for w in re.findall(pattern, logfile):
        axes[term].append(float(w[len(term):]))


args = ['learning_rate', 'cnn_learning_rate', 'cnn_weight_decay', 'finetune_cnn_after', 'learning_rate_decay_start', 'learning_rate_decay_every', 'max_iters', 'beam_size', 'rnn_size','input_encoding_size','optim', 'drop_prob_lm']

lr_decay_start = None
lr_decay_every = None

textbox = ''

for arg in args:
    if (arg == 'optim'): # string, not a float
        textbox += arg + ": " + logfile[logfile.find(arg + ": "):logfile.find("\n",logfile.find(arg + ": "))][len(arg + ": "):-1].strip() + "\n"
    else:
        pattern = re.compile(r'[ \t\n\r\f\v-]' + arg + r'[: ][ ]*' + r'[0-9eE-]+[\.]*[0-9eE-]*')
    
        search = re.search(pattern, logfile)
        if search is not None:
            #txt = 'learning_rate_decay_start'
            
            textbox += search.group(0)[1:] + "\n"
        else:
            textbox += arg + ": ????\n"

max_score = 0.0

'''Bleu_1 = 0.43140330861377
ROUGE_L = 0.28517836470595
METEOR = 0.15430470271106
Bleu_4 = 0.087902666879794
Bleu_3 = 0.14625760297104
Bleu_2 = 0.24804182869488
CIDEr = 0.048154865074749'''


scores = {"Bleu_1" : 0, "ROUGE_L" : 0, "METEOR" : 0, "Bleu_4" : 0, "Bleu_3" : 0, "Bleu_2" : 0, "CIDEr" : 0}

pattern = re.compile(r'[ \t\n\r\f\v-]' + "current_score = best_score = " + r'[0-9eE-]+[\.]*[0-9eE-]*')
search = re.findall(pattern, logfile)
if search is not None:
    for score in search:
        score_str = score[len("current_score = best_score = ") + 1:len(score)]
        curr_score = float(score_str)
        
        if curr_score > max_score:
            max_score = curr_score
            bleu_space = logfile[logfile.rfind(" **************************", 0, logfile.find(score)):logfile.find(score)] # rfind returns last index where something is found
            
            for key in scores:
                
                start = bleu_space.find((key) + " = ") + len((key) + " = ")
                nl = start
                
                while (bleu_space[nl] != "\n"):
                    nl += 1
                
                scores[key] = float(bleu_space[start:nl])
            
            #print(logfile[105859-100:105859+100])

#print(bleu_space)

#print(str(scores))

#print("Max score: " + str(max_score))

textbox += "\n"

for key in ["Bleu_1     ", "Bleu_2     ", "Bleu_3     ", "Bleu_4     ", "METEOR  ", "ROUGE_L ", "CIDEr       ",]:
    textbox += (("best " + key + " ")) + "{:.5f}".format(scores[key.strip()]) + "\n"


pyplot.plot(axes[labels[0]], axes[labels[2]])
pyplot.plot(axes[labels[0]], axes[labels[1]])

pyplot.annotate(textbox, xy=(0.2, 0.33), xycoords='axes fraction')

pyplot.title('Training vs. Validation loss from ' + filename)
pyplot.ylabel('loss')
pyplot.xlabel('iteration')
pyplot.legend(['train', 'validation'], loc='upper right')

fig = pyplot.gcf()
fig.canvas.set_window_title(filename)

pyplot.show()


