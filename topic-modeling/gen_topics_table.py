# generates a table in latex given a directory with several .topics files

import os 

import ast

BASE_DIR = "./topics"

LATEX_BEGIN = '''\\begin{table}[]
\caption{From `%s`}
\\begin{tabular}{lllll}
\\hline
Assigned Label & Top 7 Words \\\\ \\hline \\\\
'''

# format of a table line in latex
LATEX_LINE = " %s & %s & %s  \\\\"



LATEX_END = ''' &  &  \\\\ \\hline
\\end{tabular}
\\end{table}
'''



for topics_file in os.listdir(BASE_DIR):
    
    topics_list = None
    
    with open(os.path.join(BASE_DIR,topics_file)) as f:
        topics_list = list(ast.literal_eval(f.read()))
    
    # we only want the top 10 topics
    topics_list = topics_list[0:10]
    assert(len(topics_list) == 10)
    
    curr_latex_table = LATEX_BEGIN%(topics_file)
    
    for top in topics_list:
        split_words = top[1].split('"')
        top_7_words = " ".join([split_words[i] for i in range(1,len(split_words),2)])
        curr_latex_table+=(LATEX_LINE%("???",top_7_words,"???"))
    
    curr_latex_table += LATEX_END
    
    print(curr_latex_table)
    
    # now generate a LaTex table for this topics list











