# script to make a mapping from im2txt filenames like "image_id_0009185.jpg" to their corresponding screenshot ("com.kitkatandroid.keyboard-screens/screenshot_3.jpg")
# outputs the mapping (dictionary) as a json

import os
import json

ids_to_screens = {} # dictionary mapping image ids to their screenshots
                    # i.e. "0009185" -> "com.kitkatandroid.keyboard-screens/screenshot_3.jpg"

root = "im2txt-split-bg9"

for direc in os.listdir(root):
    for link in os.listdir(os.path.join(root,direc)):
        if link.find("image_id") != -1:
            image_id = link[link.rfind("image_id_")+len("image_id_"):-4] # a string like "0009185"
            
            
            splitted_path = os.readlink(os.path.join(root,direc,link)).split("/") # os.readlink(path)
            
            if (len(splitted_path) >= 2):
                screenshot = splitted_path[len(splitted_path)-2] + "/" + splitted_path[len(splitted_path)-1] # a string like "com.kitkatandroid.keyboard-screens/screenshot_3.jpg"

                ids_to_screens[image_id] = screenshot # make the mapping




out_file = open("imageids-to-screens.json", "w")

json.dump(ids_to_screens, out_file)

print("Wrote to " + out_file.name)
