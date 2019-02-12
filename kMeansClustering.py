# To run in the terminal : python3 kMeansClustering.py Set1/ (Folder which has the images)

import numpy as np
import cv2
import sys
import os

# importing library for plotting 
from matplotlib import pyplot as plt 

if(len(sys.argv) != 2):
	print("Missing arguement")

# Loading images
allImages = []
folderName = sys.argv[1]
allFiles = []
for file in os.listdir(folderName):
	img = cv2.imread(os.path.join(folderName,file), 1)
	if img is not None:
		print("File: ", file)
		allFiles.append(file)
		img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
		allImages.append(img2)

print("Total Images: ", len(allImages))	

# Gaussian Blur and Otsu Thresholding
imgOne = np.array(allImages[0])
imgBlur = cv2.GaussianBlur(imgOne,(5,5),0)
ret, imgOneThresh = cv2.threshold(imgBlur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

imgOneFlat = imgOneThresh.reshape(imgOneThresh.shape[0] * imgOneThresh.shape[1], 1) # Reshape into a column vector

#Using X, Y, Grayscale value as features
Coord = np.zeros((imgOne.shape[0] * imgOne.shape[1], 1))
for i in range(0,Coord.shape[0]):
	Coord[i][0] = i

finalVec = np.hstack((imgOneFlat, Coord))
finalVec = np.float32(finalVec)

# using K-means clustering
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
compactness,labels,centers = cv2.kmeans(finalVec,3,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Three clusters (Top, Document, Bottom)
first = finalVec[labels.ravel()==0]
second = finalVec[labels.ravel()==1]
third = finalVec[labels.ravel()==2]

print("Final Vec size: " , finalVec.shape)

# Plotting the data
plt.scatter(first[:,0],first[:,1])
plt.scatter(second[:,0],second[:,1])
plt.scatter(third[:,0],third[:,1])
plt.scatter(centers[:,0],centers[:,1],s = 50,c = 'r', marker = 's')
plt.xlabel('GrayValue'),plt.ylabel('Coordinate')
plt.show()