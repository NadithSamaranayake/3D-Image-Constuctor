import sys
import cv2
import numpy as np

def convert_negative_to_positive(image):
    return 255 - image

def histogram_equalization(image):
    if len(image.shape) == 2:  # Grayscale image
        return cv2.equalizeHist(image)
    elif len(image.shape) == 3:  # Color image
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def noise_reduction(image):
    return cv2.medianBlur(image, 5)

def edge_detection(image):
    return cv2.Canny(image, 100, 200)

def sharpen_image(image):
    gaussian_blur = cv2.GaussianBlur(image, (9, 9), 10.0)
    return cv2.addWeighted(image, 1.5, gaussian_blur, -0.5, 0)

# Read input and output paths and filter type from command line arguments
input_path = sys.argv[1]
output_path = sys.argv[2]
filter_type = sys.argv[3]

# Load the image
image = cv2.imread(input_path, cv2.IMREAD_COLOR)

# Check if the image was loaded successfully
if image is None:
    print(f"Error: Unable to open image file {input_path}")
    sys.exit(1)

# Apply the selected filter
if filter_type == 'convert_negative_to_positive':
    processed_image = convert_negative_to_positive(image)
elif filter_type == 'histogram_equalization':
    processed_image = histogram_equalization(image)
elif filter_type == 'noise_reduction':
    processed_image = noise_reduction(image)
elif filter_type == 'edge_detection':
    processed_image = edge_detection(image)
elif filter_type == 'sharpen_image':
    processed_image = sharpen_image(image)
else:
    print(f"Error: Unknown filter type {filter_type}")
    sys.exit(1)

# Save the processed image
cv2.imwrite(output_path, processed_image)
