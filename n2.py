import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# Load the negative image
image_path = './pics/negative-image.jpg'
negative_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Check if the image was loaded successfully
if negative_image is None:
    print(f"Error: Unable to open image file {image_path}")
    exit()

# Convert negative to positive
positive_image = convert_negative_to_positive(negative_image)

# Apply histogram equalization
equalized_image = histogram_equalization(positive_image)

# Reduce noise
denoised_image = noise_reduction(equalized_image)

# Detect edges
edges = edge_detection(denoised_image)

# Sharpen image
sharpened_image = sharpen_image(denoised_image)

# Display results
images = {
    "Negative": negative_image,
    "Positive": positive_image,
    "Equalized": equalized_image,
    "Denoised": denoised_image,
    "Edges": edges,
    "Sharpened": sharpened_image
}

plt.figure(figsize=(12, 8))
for i, (title, img) in enumerate(images.items()):
    plt.subplot(2, 3, i + 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()

# Save the final image
output_path = './pics/final_image.jpg'
cv2.imwrite(output_path, sharpened_image)
print(f"Final image saved to {output_path}")
