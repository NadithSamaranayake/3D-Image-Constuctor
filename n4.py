import cv2
import numpy as np
import matplotlib.pyplot as plt

def adjust_gamma(image, gamma=1):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# Read the negative image
negative_image = cv2.imread('./pics/negative-image.jpg')

# Invert the image to get the positive image
positive_image = cv2.bitwise_not(negative_image)

# Apply Gaussian Blurring to reduce noise
blurred_image = cv2.GaussianBlur(positive_image, (1, 1), 0)  # Reduce kernel size

# Apply Bilateral Filtering to preserve edges while reducing noise
bilateral_filtered_image = cv2.bilateralFilter(blurred_image, d=7, sigmaColor=75, sigmaSpace=75)  # Adjust parameters

# Sharpen the image using Unsharp Masking
gaussian_blurred = cv2.GaussianBlur(bilateral_filtered_image, (0, 0), 3)
unsharp_image = cv2.addWeighted(bilateral_filtered_image, 1.5, gaussian_blurred, -0.5, 0)

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(unsharp_image, cv2.COLOR_BGR2HSV)

# Adjust hue, saturation, and value if needed
h, s, v = cv2.split(hsv_image)
s = cv2.add(s, 7)
v = cv2.add(v, 10)
s = np.clip(s, 0, 255)
v = np.clip(v, 0, 255)
adjusted_hsv_image = cv2.merge([h, s, v])

# Convert back to BGR color space
color_corrected_image = cv2.cvtColor(adjusted_hsv_image, cv2.COLOR_HSV2BGR)

# Apply Gamma Correction with gamma less than 1 to reduce brightness
gamma_corrected_image = adjust_gamma(color_corrected_image, gamma=0.5)

# Apply Median Filtering to reduce noise
median_filtered_image = cv2.medianBlur(gamma_corrected_image, 3)  # Reduce kernel size

# Increase brightness and contrast to enhance exposure
alpha = 1.3  # Contrast control (1.0-3.0)
beta = 30    # Brightness control (0-100)
exposed_image = cv2.convertScaleAbs(median_filtered_image, alpha=alpha, beta=beta)

# Convert to LAB color space
lab_image = cv2.cvtColor(exposed_image, cv2.COLOR_BGR2LAB)

# Split LAB image to different channels
l, a, b = cv2.split(lab_image)

# Apply CLAHE to L-channel
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl = clahe.apply(l)

# Merge the CLAHE enhanced L-channel with the a and b channel
limg = cv2.merge((cl, a, b))

# Convert LAB image back to color (RGB)
equalized_color_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# Apply interpolation to resize the image
height, width = equalized_color_image.shape[:2]
interpolated_image = cv2.resize(equalized_color_image, (2 * width, 2 * height), interpolation=cv2.INTER_LINEAR)

# Apply additional sharpening to the interpolated image
final_sharpened_image = sharpen_image(interpolated_image)

#Equlizing the final image
def histogram_equalization(final_sharpened_image):
    if len(final_sharpened_image.shape) == 2:  # Grayscale image
        return cv2.equalizeHist(final_sharpened_image)
    elif len(final_sharpened_image.shape) == 3:  # Color image
        ycrcb = cv2.cvtColor(final_sharpened_image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

equalized_image = histogram_equalization(final_sharpened_image)

# Save the final converted image
cv2.imwrite('positive_image_enhanced6.jpg', final_sharpened_image)
