import cv2
import numpy as np

#Loading the negative image
negative_image = cv2.imread("./pics/champion.jpg", cv2.IMREAD_COLOR)

#Converting the negative image to a positive image
positive_image = 255 - negative_image


#Applying noise reduction techniques
#Applying Guassian Blur
blurred_image = cv2.GaussianBlur(positive_image, (5, 5), 0)

#Applying median filter
median_image = cv2.medianBlur(positive_image, 5)

#Applying bilateral filter
bilateral_image = cv2.bilateralFilter(positive_image, 9, 75, 75)

#Applying contrast enhancement techniques
# Convert to grayscale for histogram equalization
gray_image = cv2.cvtColor(positive_image, cv2.COLOR_BGR2GRAY)

# Histogram equalization
equalized_image = cv2.equalizeHist(gray_image)

# CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(gray_image)


#Applying sharpening techniques
# Laplacian filter
laplacian_image = cv2.Laplacian(positive_image, cv2.CV_64F)

# Unsharp masking
gaussian_blur = cv2.GaussianBlur(positive_image, (9, 9), 10.0)
unsharp_image = cv2.addWeighted(positive_image, 1.5, gaussian_blur, -0.5, 0)

#Saving the image
cv2.imwrite('enhanced_image2.jpg', unsharp_image)
