import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter

def downsample_image(image, scale_factor):
    """Downsamples the image by the given scale factor."""
    height, width = image.shape[:2]
    new_height, new_width = int(height / scale_factor), int(width / scale_factor)
    return cv2.resize(image, (new_width, new_height))

# Load the image
image_path = './pics/test.png'  # Replace with your image path
image = cv2.imread(image_path)

# Downsample the image to reduce memory usage
scale_factor = 4  # Adjust this factor to balance quality and performance
downsampled_image = downsample_image(image, scale_factor)

# Convert the downsampled image to grayscale for height map
gray_image = cv2.cvtColor(downsampled_image, cv2.COLOR_BGR2GRAY)

# Smooth the grayscale image to create a smoother height map
smoothed_gray_image = gaussian_filter(gray_image, sigma=2)

# Get the dimensions of the downsampled image
height, width = smoothed_gray_image.shape

# Create a mesh grid for the x and y coordinates
x = np.linspace(0, width - 1, width)
y = np.linspace(0, height - 1, height)
x, y = np.meshgrid(x, y)

# Normalize the smoothed grayscale image values for height map
z = smoothed_gray_image / 255.0

# Normalize the downsampled image for color mapping
downsampled_image_normalized = downsampled_image / 255.0

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface with original colors
ax.plot_surface(x, y, z, facecolors=downsampled_image_normalized, rstride=1, cstride=1, shade=False, antialiased=True)

# Set plot labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Height')

# Hide the axes
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Optimize the view
ax.view_init(elev=30, azim=60)
ax.dist = 8

# Show the plot
plt.show()
