import pandas as pd

'''The purpose of this code is to create parallel text files from three different representations of 
xml files and captions describing their content'''

unique_path = unique_path

seq2seq_base = "../../processed-seq2seq-uiautomator-data/"

#Files to be aligned
key_csv = seq2seq_base + "key.csv"
text_csv = seq2seq_base + "text.csv"
type_csv = seq2seq_base + "type.csv"
type_text_csv = seq2seq_base + "type_text.csv"
type_text_loc_csv = seq2seq_base + "type_text_loc.csv"





#Destination for newly created aligned files
NEW_BASE = "/Users/georgewpurnell/Desktop/new_data/"
NEW_CSV_KEY = NEW_BASE + "key.csv"
NEW_CSV_TEXT = NEW_BASE + "text.csv"
NEW_CSV_TYPE = NEW_BASE + "type.csv"
NEW_CSV_TEXT_TYPE = NEW_BASE + "type_text.csv"
NEW_CSV_TEXT_TYPE_LOC = NEW_BASE + "type_text_loc.csv"
NEW_CAPTION_FILE = NEW_BASE + "caption.csv"

files_list = [key_csv, text_csv, type_csv, type_text_csv, type_text_loc_csv]
files_content_list = []

#store the file content from the csv files in a list for usage later
for fn in files_list:
    with open(fn, "r") as f:
        lines = f.read().splitlines()
        files_content_list.append(lines)



new_files_list = [NEW_CSV_KEY, NEW_CSV_TEXT, NEW_CSV_TYPE, NEW_CSV_TEXT_TYPE, NEW_CSV_TEXT_TYPE_LOC]

FILENAME_BASE_LEN = len("http://173.255.245.197:8080/GEMMA-CP/Clarity/")
FILENAME_OFFSET = len("-screens/screenshot_5.png")

KEY_BASE = "/Users/georgewpurnell/Desktop/Clarity-Xmls/"
KEY_OFFSET = "-screens/hierarchy_1.xml"

#Create pandas dataframes from each of the csv files
key_df = pd.read_csv(CSV_KEY, names=["key"])
text_df = pd.read_csv(CSV_TEXT, names=["text"])
type_df = pd.read_csv(CSV_TYPE, names=["type"])
text_type_df = pd.read_csv(CSV_TEXT_TYPE, names=["text_type"])
text_type_loc_df = pd.read_csv(CSV_TEXT_TYPE_LOC, names=["text_type_loc"])
captions_df = pd.read_csv(unique_csv)

#Fill all unanswered descriptions with an empty string
captions_df = captions_df.fillna("")


df_list = [key_df, type_df, text_type_df, text_type_loc_df]


def string_replace(s, char, position):
    '''Helper function to replace a character in a string at a specific index'''
    return s[:position] + char + s[position+len(char):]

#arrange all files in the same order as the captions file
for i in range(len(captions_df)):
    package_name = captions_df["Filename"][i][FILENAME_BASE_LEN:-FILENAME_OFFSET]
    
    char = captions_df["Filename"][i][-FILENAME_OFFSET:] [-5]
    
    position = -5
    
    package_and_offset = package_name + string_replace(KEY_OFFSET, char, position)
    
    bool_lst = key_df["key"].str.contains(package_and_offset,case=True, regex=False)
    
    indices = key_df[bool_lst].index.values.astype(int)
    
    print(indices[0])
    
    for j in range(len(files_content_list)):
        with open(new_files_list[j], "a+") as f:
            f.write(files_content_list[j][indices[0]] + "\n")
    with open(NEW_CAPTION_FILE, "a+") as f:
        lst = list(captions_df.loc[i, 'High' : 'Low4'])
        f.write(str(lst)[1:-1] + '\n')
