import cv2
import numpy as np

def adjust_gamma(image, gamma=1):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Read the negative image
negative_image = cv2.imread('C:\\.DRIVE A\\.HDSE\\DIP\\DIP\\.COURSEWORK\\1662023355067-negate.jpg')

# Invert the image
positive_image = cv2.bitwise_not(negative_image)

# Apply Gaussian Blurring to reduce noise
blurred_image = cv2.GaussianBlur(positive_image, (5, 5), 0)

# Sharpen the image
sharpening_kernel = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]])
sharpened_image = cv2.filter2D(blurred_image, -1, sharpening_kernel)

# Interpolate the image (resizing)
scale_factor = 2
interpolated_image = cv2.resize(sharpened_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

# Convert the image to LAB color space
lab_image = cv2.cvtColor(interpolated_image, cv2.COLOR_BGR2LAB)

# Apply CLAHE to the L channel
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
lab_image[:, :, 0] = clahe.apply(lab_image[:, :, 0])

# Convert back to BGR color space
color_corrected_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

# Apply Gamma Correction with gamma less than 1 to reduce brightness
gamma_corrected_image = adjust_gamma(color_corrected_image, gamma=0.7)

# Apply Median Filtering to reduce noise
median_filtered_image = cv2.medianBlur(gamma_corrected_image, 5)

# Convert the image to YUV color space
yuv_image = cv2.cvtColor(median_filtered_image, cv2.COLOR_BGR2YUV)

# Equalize the histogram of the Y channel
yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])

# Convert back to BGR color space
final_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

# Increase brightness and contrast to enhance exposure
alpha = 2  # Contrast control (1.0-3.0)
beta = 30    # Brightness control (0-100)

exposed_image = cv2.convertScaleAbs(final_image, alpha=alpha, beta=beta)

# Save the converted image
cv2.imwrite('positive_image_enhanced.jpg', final_image)