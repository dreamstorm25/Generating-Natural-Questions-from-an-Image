import numpy as np
import imutils
import cv2
import os
import math
import operator
#from sklearn.cross_validation import train_test_split
import nltk
import csv
import re
from glob import glob
import random


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def image_to_feature_vector(image, size=(480, 480)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size)

def extract_color_histogram(image, bins=(8, 8, 8)):

    # extract a 3D color histogram from the HSV color space using
    # the supplied number of `bins` per channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
        [0, 180, 0, 256, 0, 256])

    # handle normalizing the histogram if we are using OpenCV 2.4.X
    if imutils.is_cv2():
        hist = cv2.normalize(hist)

    # otherwise, perform "in place" normalization in OpenCV 3 (I
    # personally hate the way this is done
    else:
        cv2.normalize(hist, hist)

    # return the flattened histogram as the feature vector
    return hist.flatten()

#def euclideanDistance(instance1, instance2):
#    dist = []
#    for x in range(3):
#        a = instance1[:,:,x]
#        b = instance2[:,:,x]
#        dis = np.subtract(a,b)
#        dis = np.square(dis)
#        dis = dis.mean()
#        dis = math.sqrt(dis)
#        dist.append(dis)
#
#    distance = pow(dist[0],2) + pow(dist[1],2) + pow(dist[2],2)
#    return math.sqrt(distance)

def feature_matcher(des1,des2):
    # Initiate SIFT detector
    #sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    #kp1, des1 = sift.detectAndCompute(instance1,None)
    #kp2, des2 = sift.detectAndCompute(instance2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    return len(good)


def getNeighbors(trainingSet, labels, testInstance, k):
    distances = []
    #length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = feature_matcher(testInstance, trainingSet[x])
        distances.append((labels[x], dist))
    distances.sort(key=operator.itemgetter(1),reverse=True)
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getQuestion(neighbor):

    maxscore = 0
    maxques = ""
    questions = []

    for q in neighbor:
        score = 0
        for rest in neighbor:
            if (q!=rest):

                score += nltk.translate.bleu_score.sentence_bleu([q.split(' ')], rest.split(' '))

        questions.append((score,q))
        if (score>maxscore):
            maxscore = score
            maxques = q
    questions.sort(key=operator.itemgetter(0),reverse=True)
    #print(questions)
    
    return maxques

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] is predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

path_csv1 = "C:/Users/Rishab/Documents/Major/Modified/sorted_final_coco_test"
path_csv2 = "C:/Users/Rishab/Documents/Major/Modified/sorted_final_coco_train"
path_img1 = "C:/Users/Rishab/Documents/Major/Modified/coco_test_all_download"
path_img2 = "C:/Users/Rishab/Documents/Major/Modified/coco_train_all_download"
# initialize the raw pixel intensities matrix, the features matrix,
# and labels list
rawImages_train = []
features_train = []
labels_train = []
rawImages_test = []
features_test = []
labels_test = []
i=0
display_image = []
extension = '.jpg'

j=0
k=0
with open(path_csv1 + '.csv', 'rt') as f:
    reader = csv.reader(f)
    for row in reader:
        k += 1
        if (j==0):
            j=1
            continue
        label = row[2]
        labels_test.append(label)

with open(path_csv2 + '.csv', 'rt') as f:
    reader = csv.reader(f)
    for row in reader:
        k += 1
        if (j==0):
            j=1
            continue
        label = row[2]
        labels_train.append(label)


#print (labels)
# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
#kp1, des1 = sift.detectAndCompute(instance1,None)

# loop over the input images
files_list = glob(os.path.join(path_img1,'*.jpg'))
for file in sorted(files_list, key = numericalSort):
    # load the image and extract the class label (assuming that our
    # path as the format: /path/to/dataset/{class}.{image_num}.jpg

    ext = os.path.splitext(file)[-1].lower()
   # print(ext)
    if ext!=extension:
        continue

    i+=1
    #print(file)

    #if (i>200):
        #break

    #print(file)
    image = cv2.imread(file, 0)
    image = cv2.resize(image,(480,480))
    display_image.append(image)
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image,None)

    
    image_id = file.split(os.path.sep)[-1].split(".")[0]



    #if (row%100 ==0): print(label)


    # extract raw pixel intensity "features", followed by a color
    # histogram to characterize the color distribution of the pixels
    # in the image
    #pixels = image_to_feature_vector(image)
    #hist = extract_color_histogram(image)

    # update the raw images, features, and labels matricies,
    # respectively
    rawImages_test.append(des1)
    #features.append(hist)
    #labels.append(label)

files_list = glob(os.path.join(path_img2,'*.jpg'))
for file in sorted(files_list, key = numericalSort):
    # load the image and extract the class label (assuming that our
    # path as the format: /path/to/dataset/{class}.{image_num}.jpg

    ext = os.path.splitext(file)[-1].lower()
   # print(ext)
    if ext!=extension:
        continue

    i+=1
    #print(file)

    #if (i>200):
        #break

    #print(file)
    image = cv2.imread(file, 0)
    image = cv2.resize(image,(480,480))
    

    image_id = file.split(os.path.sep)[-1].split(".")[0]
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image,None)




    #if (row%100 ==0): print(label)


    # extract raw pixel intensity "features", followed by a color
    # histogram to characterize the color distribution of the pixels
    # in the image
    #pixels = image_to_feature_vector(image)
    #hist = extract_color_histogram(image)

    # update the raw images, features, and labels matricies,
    # respectively
    rawImages_train.append(des1)
    #features.append(hist)
    #labels.append(label)


#rawImages = np.array(rawImages)
#features = np.array(features)
#labels_train = np.array(labels_train)
#labels_test = np.array(labels_test)
#(trainRI, testRI, trainRL, testRL) = train_test_split(rawImages, labels, test_size=0.1, random_state=42)

trainRI = rawImages_train
testRI = rawImages_test
trainRL = labels_train
testRL = labels_test


#(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(features, labels, test_size=0.25, random_state=42)

#print(trainRI.size)

i = 0
for img in testRI:
    neighbors = getNeighbors(trainRI, trainRL, img, 15)

    questions = []

    for n in neighbors:
        ques = getQuestion(n.split("---"))
        questions.append(ques)

    #print("final")
    final = getQuestion(questions)

    print(final)
    #answer = testRL[i].split("---")
    #x = random.randrange(0, 50, 1)
    #print(x)
    #y = random.randint(0, 4)
    #if(x%3==0):
    #    print(answer[y])
    #else:
    #    print(final)
        #print(testRL[i])
    cv2.imshow('image',display_image[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    i += 1

    input("Press Enter")


'''

model = KNeighborsClassifier(n_neighbors=5)
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

# train and evaluate a k-NN classifer on the histogram
# representations
print("[INFO] evaluating histogram accuracy...")
model = KNeighborsClassifier(n_neighbors=5)
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))


# construct the argument parse and parse the arguments

#ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--dataset", required=True, help="gives path to image")
#ap.add_argument("-k", "--neighbors", type=int, default=1, help="# of nearest neighbors for classification")
#ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of jobs for k-NN distance (-1 uses all available cores)")

#args = vars(ap.parse_args())
#print (args['dataset'])

#imagePaths = list(paths.list_images(args["dataset"]))
'''
