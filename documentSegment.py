# To run in the terminal : python3 documentSegment.py Set1 (Folder which contains images)

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
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
		allImages.append(img)

print("Total Images: ", len(allImages))	

# Finding the four corner points of the cropped document
i = 0
for inputImg in allImages:
	imgBlur = cv2.GaussianBlur(inputImg,(5,5),0)
	ret, imgThresh = cv2.threshold(imgBlur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	print("Shape of the Image: ", inputImg.shape)

	for r in range(imgThresh.shape[0]):
		if(imgThresh[r][0] == 255):
			point1 = (r,0)
			break

	for r in range(imgThresh.shape[0]):
		if(imgThresh[r][imgThresh.shape[1] - 1] == 255):
			point2 = (r,imgThresh.shape[1] - 1)
			break

	for r in range(imgThresh.shape[0] - 1, 0, -1):
		if(imgThresh[r][0] == 255):
			point3 = (r,0)
			break

	for r in range(imgThresh.shape[0] - 1, 0, -1):
		if(imgThresh[r][imgThresh.shape[1] - 1] == 255):
			point4 = (r,imgThresh.shape[1] - 1)
			break

	ylow = min(point1[0], point2[0])
	yhigh = max(point3[0], point4[0])

	finalImg = inputImg[ ylow: yhigh, 0 : inputImg.shape[0]] # Final cropped image

	# Saving the image
	name, ext = os.path.splitext(allFiles[i])
	fileName = name + 'cropped.jpg'

	cv2.imwrite(os.path.join(folderName , fileName), finalImg)
	i = i + 1

