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
for file in os.listdir(folderName):
	img = cv2.imread(os.path.join(folderName,file), 1)
	if img is not None:
		print("File: ", file)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
		allImages.append(img)

print("Total Images: ", len(allImages))	

imgBlur = cv2.GaussianBlur(allImages[0],(5,5),0)
ret, imgThresh = cv2.threshold(imgBlur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("Shape of the Image: ", allImages[0].shape)

# Trying to flood fill the document to segment it - Not working
h, w = imgThresh.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)

for i in range(imgThresh.shape[1]):
	if(imgThresh[0][i] == 255):
		cv2.floodFill(imgThresh, mask, (i,0), 255)
	if(imgThresh[imgThresh.shape[0] - 1][i] == 255):
		cv2.floodFill(imgThresh, mask, (i,(imgThresh.shape[0] - 1)), 255)

for i in range(imgThresh.shape[0]):
	if(imgThresh[i][0] == 255):
		cv2.floodFill(imgThresh, mask, (0,i), 255)
	if(imgThresh[i][imgThresh.shape[1] - 1] == 255):
		cv2.floodFill(imgThresh, mask, ((imgThresh.shape[1] - 1),0), 255)

plt.subplot(121), plt.imshow(imgThresh, 'gray')
plt.show()

