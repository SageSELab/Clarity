# Written by Ali Yachnes

# python script to read in an im2txt inference json that looks like the following:

'''

[
    {
        "image_id": "./split-sciclone/test/image_id_0009593.jpg",
        "caption1": "in the middle of the screen there is a button that lets the user exit the screen",
        "caption2": "in the middle of the screen there is a button that lets the user exit the current screen",
        "caption3": "N/A"
    },
    {
        "image_id": "./split-sciclone/test/image_id_0009772.jpg",
        "caption1": "in the center of the screen is a text field where the user inputs their email address",
        "caption2": "in the middle of the screen there is a button that lets the user exit the screen",
        "caption3": "in the middle of the screen there is a button that lets the user exit the current screen"
    }
]

'''

# and then produce a vis.json that can be used with the html/javascript files in this folder



import os
import json
import csv

from shutil import copyfile

if len(os.sys.argv) < 3:
    print("Script to take either an im2txt, ntk2, or seq2seq prediction-score json")
    print("and produce a folder (automatically named based on the input json) containing")
    print("the html and the javascript needed to visualize predictions.")
    print("usage: python " + __file__ + " <gt, im2, ntk, seq> <inference json/txt/csv>")
    print("i.e. python " + __file__ + " im2 inferences-low.json")
    print("")
    exit()

# make sure provided model type is correct

model_type = os.sys.argv[1]

assert(((model_type == "im2") or (model_type == "ntk") or (model_type == "seq") or (model_type == "gt")))

# open given inference json

inference_json = None

if model_type == "im2" or model_type == "ntk":
    inference_json = json.load(open(os.sys.argv[2], "r"))
else: # seq2seq gets read normally (without json load; not a json)
    inference_json = open(os.sys.argv[2], "r")


# we do something separate for groundtruth
if (model_type != "gt"):
    vis_template = "./vis-template"

    # copy the contents of vis-template to a new directory

    # first make a directory

    new_dir = os.path.join(".","vis-%s-%s"%(model_type,os.sys.argv[2][0:os.sys.argv[2].find(".json")]))

    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    else:
        print("Warning: directory already exists; files may be overwritten!")
        
        
    for fil in os.listdir(vis_template):
        # src, dest
        src = os.path.join(vis_template,fil)
        dest = os.path.join(new_dir,fil)
        copyfile(src, dest)
        print("Copied %s to %s"%(src, dest))

    # vis file handle
    output_vis_file = open(os.path.join(new_dir,"vis.json"), "w") 


    # where the vis json will be written to be later dumped
    vis_json = [] 

    # Note: the vis json should look like this:
    '''

    [
        {
            "caption": "below the message box there is a text field where the user can input their name",
            "path": "com.remitly.androidapp-screens/screenshot_5.jpg"
        },
        {
            "caption": "in center of the screen alert message is given",
            "path": "com.aco.cryingbebe-screens/screenshot_1.jpg"
        }
    ]

    '''

# im2txt
if model_type == "im2":

    # open im2txt dictionary which looks like this

    '''

    {
        "0010190": "com.baynews9.baynews9plus-screens/screenshot_3.jpg",
        "0010191": "com.medibang.android.colors-screens/screenshot_3.jpg",
        "0010192": "biz.binarysolutions.signature-screens/screenshot_2.jpg",
        "0010193": "com.oldguide.inter.tipstekken-screens/screenshot_1.jpg",
        "0010194": "com.androbaby.firstcolorsforbaby-screens/screenshot_1.jpg",
        "0010195": "com.dinixe.eyesmakeups-screens/screenshot_2.jpg",
        "0010196": "com.proj.minecraftpixelmonskins-screens/screenshot_1.jpg",
        "0010197": "com.ehawk.antivirus.applock.wifi-screens/screenshot_6.jpg",
        "0010198": "com.crazygame.inputmethod.keyboard7-screens/screenshot_2.jpg",
        "0010199": "com.momentgarden-screens/screenshot_5.jpg",
        "0010200": "com.futbolx.ligamx-screens/screenshot_3.jpg",
        "0010201": "com.mapswithme.maps.pro-screens/screenshot_1.jpg",
        "0010202": "com.narvii.amino.x242238269-screens/screenshot_1.jpg",
        "0010203": "com.inome.android-screens/screenshot_6.jpg",
        "0010204": "com.kitkatandroid.keyboard-screens/screenshot_3.jpg"
    }

    '''

    im2txt_dict = json.load(open("imageids-to-screens.json", "r"))


    # im2txt json should look like this:
    
    '''
    
    {
        "captions": [
            "in the center of the screen is a text field where the user inputs their email address",
            "in the center of the screen there is a text field where the user can input their email",
            "in the middle of the screen there is a button that lets the user exit the app"
        ],
        "image_id": "0009187",
        "scores": [
            {
                "Bleu_1": 0.588,
                "Bleu_2": 0.429,
                "Bleu_3": 0.29,
                "Bleu_4": 0.205
            },
            {
                "Bleu_1": 0.556,
                "Bleu_2": 0.362,
                "Bleu_3": 0.201,
                "Bleu_4": 0.0
            },
            {
                "Bleu_1": 0.647,
                "Bleu_2": 0.493,
                "Bleu_3": 0.401,
                "Bleu_4": 0.343
            }
        ]
    },
    
    '''
    

    for entry in inference_json:
        
        
        # old code:
        #image_id = entry["image_id"].split("/")[len(entry["image_id"].split("/"))-1] 
        #image_id = image_id[len("image_id_"):image_id.rfind(".")]
        
        image_id = entry["image_id"]
        
        vis_entry = {}
        
        vis_entry["path"] = im2txt_dict[image_id]
        
        vis_entry["caption"] = ""

        i = 0
        for cap in entry["captions"]:
            if cap != "":
                vis_entry["caption"] += str(i+1) + ". " + cap + "  ||  "
                i += 1


        vis_json.append(vis_entry)

    


# neuraltalk2
elif model_type == "ntk":


    # ntk2 json should look like this:
    
    '''
    
    {
        "captions": [
            "in the top left there is a back button to return the user to the previous screen",
            "in the center of the screen is a text field where the user inputs their email address",
            "in the center of the screen is a text field where the user inputs their email address"
        ],
        "scores": [
            {
                "Bleu_1": 0.444,
                "Bleu_2": 0.28,
                "Bleu_3": 0.17,
                "Bleu_4": 0.0
            },
            {
                "Bleu_1": 0.166,
                "Bleu_2": 0.099,
                "Bleu_3": 0.0,
                "Bleu_4": 0.0
            },
            {
                "Bleu_1": 0.166,
                "Bleu_2": 0.099,
                "Bleu_3": 0.0,
                "Bleu_4": 0.0
            }
        ],
        "screenshot": "com.hollar.android-screens/screenshot_1.jpg"
    },
    
    '''
    

    for entry in inference_json:

        vis_entry = {}
        
        vis_entry["path"] = entry["screenshot"]
        
        vis_entry["caption"] = ""

        i = 0
        for cap in entry["captions"]:
            if cap != "":
                vis_entry["caption"] += str(i+1) + ". " + cap + " || "
                i += 1


        vis_json.append(vis_entry)

    


# seq2seq
elif model_type == "seq":

    # seq2eq file should look like this:
    
    '''  
    the bottom of the screen there is a button to the
    the bottom of the screen there is a button for the user to go to
    the bottom of the screen there is a button to the
    the bottom of the screen there is a button to the
    the bottom of the screen there is a button to the   
    '''
    
    
    seq2seq_lines = inference_json.read().splitlines()
    
    key_file = None
    
    
    
    if inference_json.name.find("low") != -1:
        key_file = open("key-test-low.csv")
    elif inference_json.name.find("high") != -1:
        key_file = open("key-test-high.csv")
    elif inference_json.name.find("both") != -1:
        key_file = open("key-test-both.csv")
    else:
        assert(False) # error!
    
    # key_file looks like this:
    
    '''
    com.leafgreen.teen-screens/hierarchy_1.xml
    com.ToDoReminder.gen-screens/hierarchy_1.xml
    com.learninga_z.onyourown-screens/hierarchy_1.xml
    com.chic.blacklight-screens/hierarchy_3.xml
    com.peakpocketstudios.atmosphere-screens/hierarchy_1.xml
    com.tplink.skylight-screens/hierarchy_4.xml
    com.whaleshark.retailmenot-screens/hierarchy_3.xml
    com.michaels.michaelsstores-screens/hierarchy_4.xml
    '''
    
    key_lines = key_file.read().splitlines()
    
    for i, line in enumerate(seq2seq_lines):
        
        
        components = key_lines[i].split("/")
        path = components[0] + "/" + components[1][0:len(components[1])-4].replace("hierarchy_","screenshot_") + ".png"
        path = path.strip()
        
        vis_entry = {}
        
        vis_entry["path"] = path
        
        vis_entry["caption"] = line

        vis_json.append(vis_entry)

elif model_type == "gt":
    
    
    for lvl in ["high","low"]:
        vis_template = "./vis-template"

        # copy the contents of vis-template to a new directory

        # first make a directory

        new_dir = os.path.join(".","vis-groundtruth-%s"%(lvl))

        if not os.path.isdir(new_dir):
            os.mkdir(new_dir)
        else:
            print("Warning: directory already exists; files may be overwritten!")
            
            
        for fil in os.listdir(vis_template):
            # src, dest
            src = os.path.join(vis_template,fil)
            dest = os.path.join(new_dir,fil)
            copyfile(src, dest)
            print("Copied %s to %s"%(src, dest))

        # vis file handle
        output_vis_file = open(os.path.join(new_dir,"vis.json"), "w") 


        # where the vis json will be written to be later dumped
        vis_json = [] 

        # Note: the vis json should look like this:
        '''

        [
            {
                "caption": "below the message box there is a text field where the user can input their name",
                "path": "com.remitly.androidapp-screens/screenshot_5.jpg"
            },
            {
                "caption": "in center of the screen alert message is given",
                "path": "com.aco.cryingbebe-screens/screenshot_1.jpg"
            }
        ]

        '''
                
        gt_csv = open(os.sys.argv[2])
                    
        lines = gt_csv.read().splitlines()
                    
        gt_csv.close()



        for i, row in enumerate(csv.reader(lines, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)):
            
            # skip the header
            if i != 0:
                
                vis_entry = {}
                    
                vis_entry["path"] = row[0][len("http://173.255.245.197:8080/GEMMA-CP/Clarity/"):]     
                
                vis_entry["caption"] = ""
                
                if lvl == "high":
                    vis_entry["caption"] = (row[1] if row[1] != "" else "N/A; blank")
                elif lvl == "low":
                    vis_entry["caption"] = ""
                    
                    for j in range(2,5+1):
                        vis_entry["caption"] += "%d. %s || "%(j-1, row[j])
                vis_json.append(vis_entry)
                
                
        json.dump(vis_json, output_vis_file) # write out the vis json as promised
        output_vis_file.close()
        
        print("Successfully wrote to " + output_vis_file.name)
        
if model_type != "gt":
    json.dump(vis_json, output_vis_file) # write out the vis json as promised
    output_vis_file.close()

    print("Successfully wrote to " + output_vis_file.name)




