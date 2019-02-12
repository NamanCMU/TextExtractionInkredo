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

count = 0
for inputImg in allImages:

	print("Processing Image ", count + 1);
	imgOne = np.array(inputImg)
	
	# Gaussian Blur and Otsu Thresholding
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
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10,1)
	compactness,labels,centers = cv2.kmeans(finalVec,3,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

	# Three clusters
	first = finalVec[labels.flatten() == 0]
	second = finalVec[labels.flatten() == 1]
	third = finalVec[labels.flatten() == 2]

	# finding which label corresponds to the document
	# Assumption: cluster center with highest gray value (closer to white) is the correct cluster
	graycenter = np.array([centers[0][0], centers[1][0], centers[2][0]])

	if(graycenter.argmax(axis = 0) == 0):
		contentLabels = first;
	if(graycenter.argmax(axis = 0) == 1):
		contentLabels = second;
	if(graycenter.argmax(axis = 0) == 2):
		contentLabels = third;

	# Filling the document region with white
	for index in contentLabels[:,1]:
		imgOneFlat[int(index)][0] = 255 

	croppedImg = imgOneFlat.reshape(imgOneThresh.shape[0], imgOneThresh.shape[1]) & imgOne # Reshaping and then ANDing with the original image

	# Saving the image
	name, ext = os.path.splitext(allFiles[count])
	fileName = name + 'cropped.jpg'

	cv2.imwrite(os.path.join(folderName , fileName), croppedImg)

	# Plotting the data
	# plt.subplot(111), plt.imshow(croppedImg, 'gray')

	# plt.scatter(first[:,1],first[:,0], c = 'g')
	# plt.scatter(second[:,1],second[:,0], c = 'b')
	# plt.scatter(third[:,1],third[:,0], c = 'k')
	# plt.scatter(centers[:,1],centers[:,0],s = 50,c = 'r', marker = 's')
	# plt.xlabel('Coordinate'),plt.ylabel('GrayValue')
	# plt.show()
	
	count = count + 1
