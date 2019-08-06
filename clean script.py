import csv
import cv2
import operator
import pandas as pd
import numpy as np
import os

extension = '.jpg'
path_csv = "C:/Users/Rishab/Documents/Major/Dataset/bing_train_all"
path_img = "C:/Users/Rishab/Documents/Major/bing_train_all"

with open('sorted_train.csv', 'wt') as out:
    write = csv.writer(out)

    for file in os.listdir(path_img):
        # load the image and extract the class label (assuming that our
        # path as the format: /path/to/dataset/{class}.{image_num}.jpg

        ext = os.path.splitext(file)[-1].lower()
        if ext!=extension:
            continue

        image_id = file.split(os.path.sep)[-1].split(".")[0]

        print("img: ", image_id)

        with open('bing_train_all.csv', 'rt') as inp:

            for row in csv.reader(inp):
               # print(row[0])

                if row[0] == image_id:
                    #print(row[0])
                    #print(image_id)
                    write.writerow(row)




with open('sorted_train.csv', 'rt') as f:
    reader = csv.reader(f)
    sortedlist = sorted(reader, key=operator.itemgetter(0), reverse=False)

np.random.seed(0)

df = pd.DataFrame(sortedlist, columns = ['image_id', 'image_url', 'query_term', 'questions', 'captions'])

df.to_csv('sorted_final_train.csv', index= False)
