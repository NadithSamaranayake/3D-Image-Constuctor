import cv2
import numpy as np

def adjust_gamma(image, gamma=1):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Read the negative image
negative_image = cv2.imread('./pics/johnny.png')

# Invert the image to get the positive image
positive_image = cv2.bitwise_not(negative_image)

# Apply Gaussian Blurring to reduce noise
blurred_image = cv2.GaussianBlur(positive_image, (5, 5), 0)

# Apply Bilateral Filtering to preserve edges while reducing noise
bilateral_filtered_image = cv2.bilateralFilter(blurred_image, d=9, sigmaColor=75, sigmaSpace=75)

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
median_filtered_image = cv2.medianBlur(gamma_corrected_image, 5)

# Increase brightness and contrast to enhance exposure
alpha = 1.3  # Contrast control (1.0-3.0)
beta = 30    # Brightness control (0-100)
exposed_image = cv2.convertScaleAbs(median_filtered_image, alpha=alpha, beta=beta)

# Save the converted image
cv2.imwrite('positive_image_enhanced.jpg', exposed_image)

#must add interpolation and histrogram equilization and sharpen image to this code