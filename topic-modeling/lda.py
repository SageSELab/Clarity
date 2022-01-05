# script to perform LDA topic modeling on the collected corpus of captions

import csv
import nltk
import gensim
import os


if len(os.sys.argv) < 3:
    print("Script to train an LDA model on a given caption type with a given number of topics.")
    print("\nusage: python3 " + __file__ + " <high, low> <NUM_TOPICS>")
    print("ex: python3 " + __file__ + " high 15")
    exit()

train_type = os.sys.argv[1]

if train_type not in ["low", "high"]:
    print("Invalid train type `%s`; must be `high` or `low`."%(train_type))
    exit()

try:
        
    NUM_TOPICS = int(os.sys.argv[2])

    if NUM_TOPICS <= 0:
        print("Invalid NUM_TOPICS `%d`"%(NUM_TOPICS))
        exit()
except BaseException:
    print("Invalid integer `%s`"%(os.sys.argv[2]))
    exit()
    

# list of all the documents on which we perform LDA
# a document in our case is a screenshot with all of its captions (or just high, or just low)

all_documents = [] # list of strings
all_tokenized_documents = [] # list of lists

unique_csv = open("unique.csv","r")
unique_csv_lines = unique_csv.read().splitlines()
unique_csv.close()

# get a list of english stop words

en_stop_txt = open("stanford_stop_words.txt")

# en_stop_words is a list of strings; i.e. ['who', "who'd", "who'll", "who's", 'whod']

en_stop_lines = en_stop_txt.read().splitlines()

# build a dictionary of stop words for fast lookup
en_stop_words = {}
for line in en_stop_lines:
    if line.strip() != "":
        en_stop_words[line] = 1
        
en_stop_txt.close()

# construct a dictionary of alphabetical characters for fast lookup
alpha_chars = {}
for c in "abcdefghijklmnopqrstuvwxyz ":
    alpha_chars[c] = 1

# we train a separate model for high and low
num_caps = 0
# iterate through key csv row by row (HIT by HIT)
for i, row in enumerate(csv.reader(unique_csv_lines, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)):
    if i != 0: # if this row is not the header
        # order of headers in unique.csv is "Filename","High","Low1","Low2","Low3","Low4","Split"
        # so row[1] to row[5] are the actual captions
        
        if train_type == "high":
            document_captions = [row[1]]
        elif train_type == "low":
            document_captions = [row[2], row[3], row[4], row[5]]
            
        for j in range(1,5+1):
            if row[j].strip() != "":
                num_caps+=1;
        #input(str((train_type,document_captions)))
        
        # join all the captions with a space, since each screenshot needs a single document string
        # also get rid of trailing and leading whitespace in the document
        # also to_lower for good measure (even though the corpus is already lowercase)
        all_documents.append((" ".join(document_captions)).strip().lower())

print(num_caps)
exit()
print("Loaded all raw documents with train_type `%s`!"%(train_type))



for i, raw_doc in enumerate(all_documents):
    
    filtered_doc = ""
    # filter out all non alphabetical characters from each raw doc (don't filter out spaces)
    for c in raw_doc:
        if c in alpha_chars:
            filtered_doc += c
    
    all_documents[i] = filtered_doc

    
print("Filtered out all non-alphabetical characters from the raw documents!")

# now tokenize each document and put it into all_tokenized_documents

for doc in all_documents:
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    # append the tokenized version of this document to all_tokenized_documents
    all_tokenized_documents.append(tokenizer.tokenize(doc))


print("Tokenized each document!")

# now remove stop words from each tokenized document and replace its entry in all_tokenized_documents

for i, tokenized_doc in enumerate(all_tokenized_documents):
    all_tokenized_documents[i] = [word for word in tokenized_doc if word not in en_stop_words]

print("Removed stop words from each document!")


# now perform stemming; this is mapping similar words to a single stemp (i.e. running, runner, ran map to run)


stemmer = nltk.PorterStemmer()
for i, tokenized_doc in enumerate(all_tokenized_documents):
    #print(all_tokenized_documents[i])
    all_tokenized_documents[i] = [stemmer.stem(word) for word in tokenized_doc]
    #print(all_tokenized_documents[i])

print("Stemmed each document!")


# construct a document term matrix

dictionary = gensim.corpora.Dictionary(all_tokenized_documents)

# dictionary.token2id is a dictionary giving us the mapping of words to their id; i.e. 'fast': 240

# corpus is a list of lists of tuples like [(0, 2), (1, 3), (2, 3)]
# where each tuple is (term_ID, term_count)
corpus = [dictionary.doc2bow(tokens) for tokens in all_tokenized_documents]

print("Generated dictionary and corpus for all documents!")


# construct an LDA for each NUM_TOPICS we're interested in

# the number of topics for the lda model varies each iteration; we train 4 LDA models


print("Training LDA with NUM_TOPCS = %d..."%(NUM_TOPICS))

# the top `TOP_N_WORDS` words will be printed for each of the `NUM_TOPICS` topics
TOP_N_WORDS = 7

# the more passes the model makes, the more accurate it is (and the higher the compute time)
# i.e. number of epochs
NUM_PASSES = 200

# 500 iterations were suggested in the issue
NUM_ITERATIONS = 500

# number of cores on which to train LDA
NUM_CORES = 16

LDA_MODEL = gensim.models.ldamulticore.LdaMulticore(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=NUM_PASSES, iterations=NUM_ITERATIONS, workers=NUM_CORES)

print("Trained LDA for `%s` corpus with NUM_TOPICS = %d!"%(train_type, NUM_TOPICS))


# after generating an lda model, we can get the topics like so
# the line below is an example of the output expected by print_topics;
# each float * represents
# [(0, '0.137*"button" + 0.109*"screen" + 0.101*"user" + 0.042*"left" + 0.029*"app" + 0.027*"play"')]
# we print all the topics (-1)
print()

topics_file = open(("lda-%s-%d-%d-%d.topics"%(train_type,NUM_TOPICS,NUM_ITERATIONS,NUM_PASSES)), "w")

# print_topics with -1 will order topics by significance
# also, the TOP_N_WORDS are ordered by significance
topics_file.write(str(LDA_MODEL.print_topics(num_topics=-1, num_words=TOP_N_WORDS)))
topics_file.close()

print("Successfully wrote to `%s`!"%(topics_file.name))


# now save the LDA model checkpoint

ckpt_name = "lda-%s-%d-%d-%d.ckpt"%(train_type,NUM_TOPICS,NUM_ITERATIONS,NUM_PASSES)
LDA_MODEL.save(ckpt_name)

print("Successfully wrote to `%s`!"%(ckpt_name))
