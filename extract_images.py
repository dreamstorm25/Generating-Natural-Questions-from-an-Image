import csv
import os
from urllib import request
import requests
#path where all csv files are located
path = "C:/Users/Rishab/Documents/Major/Dataset"
for file in os.listdir(path):
    i = 1
    ext = os.path.splitext(file)
    #create a new folder for each csv file, this line is misplaced should be after if so make sure you don't have any other type files in the folder keep this script also seperate for precaution
    if not os.path.exists(ext[0]+"_download"):
        os.makedirs(ext[0]+"_download")
    #f = open(ext[0]+"_download", 'wb') this will not work in windows, it may work in mac
    if ext[1] == ".csv":
        with open(os.path.join(path,file)) as file_obj:
            reader = csv.DictReader(file_obj)
            for line in reader:
                try:
                    url = line["image_url"]
                    #f.write(request.urlopen(url).read())
                    img_data = requests.get(url, timeout=20).content
                    with open(ext[0]+"_download/" + line["image_id"] +".jpg",'wb') as handler:
                        handler.write(img_data)
                except Exception as e:
                    print(e)
                    print(line["image_id"])
                #count number of images downloaded
                i = i+1

