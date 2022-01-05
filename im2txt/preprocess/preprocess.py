import numpy as np
import pandas as pd
import argparse
#import requests
import os
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("id", help="Image id number. Must be 7 digits. " +
                      "If the number is less than 7 digits use leading zeros. " +
                      "Ex. 1 would be 0000001")
    parser.add_argument("--in", dest="csv_fldr_path", help="Path of the csv file to process")
    parser.add_argument("--out", dest="out_path", help="Path of the folder to save images in")
    parser.add_argument("--annotations", dest="annotations", help="Path of the folder to save caption json files in")
    args = parser.parse_args()

    #caption_count keeps track of the caption id
    caption_count = 0



    #source = "/source_dir"
    train_dest = args.out_path + "/train"
    val_dest = args.out_path + "/val"
    test_dest = args.out_path + "/test"
    #files = os.listdir(source)


    #counter keeps track of the image id
    counter = int(args.id)
    imgs = args.csv_fldr_path

    lst = [ imgs + "/" + f for f in os.listdir(imgs) if f[-3:] == "csv"]

    train_json = {}
    val_json = {}

    train_image_lst = []
    val_image_lst = []
    test_image_lst = []

    train_caption_lst = []
    val_caption_lst = []
    test_caption_lst = []


    """def download_and_write_images_to_dir(dir, url, img_id):
        Download and write images to given directory.

        Arguments:
        dir -- the directory to write to
        url -- the url of the image to download
        img_id -- the file will be named using the image id
        
        
        img = requests.get(url).content
        with open(dir + "/image_id_" + img_id + ".jpg", 'wb') as handler:
            handler.write(img)"""


    def make_caption_list(df):
        """ Make a list of the image captions. 

        Each entry in the list is a pandas series 
        object containing a type of caption 
        ('high level', 'low_level1' etc)
        
        Arguments:
        df -- a pandas dataframe containing caption columns
        """
        
        high_level = (list(df["Answer.HighLevel"]))
        low_level_1 = (list(df["Answer.LowLevel1"]))
        low_level_2 = (list(df["Answer.LowLevel2"]))
        low_level_3 = (list(df["Answer.LowLevel3"]))
        low_level_4 = (list(df["Answer.LowLevel4"]))
        return [high_level,low_level_1, low_level_2, low_level_3, low_level_4]


    def make_image_json_entry(img_id, height, width):
        """Return a dictionary that can be stored as a json

        Arguments:
        img_id -- image id     -> string
        height -- image height -> string
        width  -- image width  -> string
        """
        image_entry = {}
        image_entry["file_name"] = "image_id_" + img_id + ".jpg"
        image_entry["height"] = height
        image_entry["width"] = width
        image_entry["id"] = img_id
        return image_entry


    def make_caption_json_entry(img_id, caption_id, caption):
        """Return a dictionary that can be stored as a json

        Arguments:
        img_id     -- the associated image id  -> string
        caption_id -- the id of the caption    -> string
        caption    -- the caption to be stored -> string
        """
        caption_entry = {}
        caption_entry["image_id"] = img_id
        caption_entry["id"] = caption_id
        caption_entry["caption"] = caption
        return caption_entry


    def append_to_caption_lst(caption_lst, img_id, caption_id, img_captions, curr_img_num):
        """Append caption entry to the caption list
        
        Arguments:
        caption_lst -- a list to hold captions    --list
        img_id      -- id of the associated image --string
        caption_id  -- id of the caption          --string
        img_captions -- captions for an image     --string
        curr_img_num -- current highest image id  --int
        """ 
        for j in range(len(img_captions)):
            #loop through the list of caption
            caption_entry = make_caption_json_entry(img_id, caption_id, img_captions[j][curr_img_num])
            caption_lst.append(caption_entry)
            caption_id += 1


    for f in lst:
        df = pd.read_csv(f)
        imgs = (df["Input.image_url"])
        #each entry of captions is a series object
        captions = make_caption_list(df) # returns [high_level,low_level_1, low_level_2, low_level_3, low_level_4]
        for i in range(len(imgs)):
            img_id = str(counter).zfill(7)

            if np.random.rand(1) < 0.2:
                download_and_write_images_to_dir(val_dest, imgs[i], img_id)
                image_entry = make_image_json_entry(img_id, "1920", "1200")
                val_image_lst.append(image_entry)
                append_to_caption_lst(val_caption_lst,img_id,caption_count,captions, i)
            #elif np.random.rand(1) < 0.3:
                #download_and_write_images_to_dir(test_dest, imgs[i], img_id)
                #append_to_caption_lst(test_caption_lst,img_id,caption_count,captions, i)
            else:
                download_and_write_images_to_dir(train_dest, imgs[i], img_id)
                image_entry = make_image_json_entry(img_id, "1920", "1200")
                train_image_lst.append(image_entry)
                append_to_caption_lst(train_caption_lst,img_id,caption_count,captions, i)
            caption_count += len(captions)
            counter += 1

    header_info = {"description": "This is the Clarity dataset.", "version": "1.0", "year": "2018"}

    train_json = {"info":header_info, "image":train_image_lst, "annotations":train_caption_lst}
    val_json = {"info":header_info, "image":val_image_lst, "annotations":val_caption_lst}

    with open(args.annotations + '/captions_train.json', 'w') as outfile:
        json.dump(train_json, outfile)

    with open(args.annotations + '/captions_val.json', 'w') as outfile:
        json.dump(val_json, outfile)

