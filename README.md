# Generating-Natural-Questions-from-an-Image
The data set has been taken from the site : https://www.microsoft.com/en-us/download/details.aspx?id=53670.
This project is an attempt to implement the approach mentioned in the paper : https://arxiv.org/abs/1603.06059.
In this project I have only implemented a retrieval model mentioned in the paper.
The extract_images.py script downloads images from the given url of the images in the dataset.
clean script.py cleans the images and organizes them.
knn2 is the main implmentation, in the project I have used sift to extract features of an image and use them to compares them with another image.
