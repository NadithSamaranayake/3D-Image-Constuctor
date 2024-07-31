import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from skimage import exposure, img_as_ubyte
from scipy.ndimage import gaussian_filter

# Read the data from the CSV file
data = pd.read_csv("./Datasets/cube_data.csv")

# Extract the X, Y, Distance (Z) coordinates
x = data['X'].values
y = data['Y'].values
z = data['Distance'].values

# Define the grid size for interpolation
grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]

# Interpolate the Z values to create a grid
grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

# Normalize the interpolated distance values to fit into the range of 0-255 for image representation
grid_z_normalized = (grid_z - np.nanmin(grid_z)) / (np.nanmax(grid_z) - np.nanmin(grid_z)) * 255
grid_z_normalized = np.nan_to_num(grid_z_normalized).astype(np.uint8)

# Apply noise reduction using Gaussian filter
grid_z_smoothed = gaussian_filter(grid_z_normalized, sigma=1)

# Enhance the image using contrast stretching
grid_z_enhanced = exposure.rescale_intensity(grid_z_smoothed, in_range='image', out_range=(0, 255))

# Plot the raw depth map and the enhanced depth map
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(grid_z_normalized, cmap='gray', origin='lower')
ax[0].set_title('Raw Depth Map')
ax[0].axis('off')

ax[1].imshow(grid_z_enhanced, cmap='gray', origin='lower')
ax[1].set_title('Enhanced Depth Map')
ax[1].axis('off')

plt.show()
