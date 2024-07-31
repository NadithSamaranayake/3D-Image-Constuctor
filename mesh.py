import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from skimage import exposure
from mpl_toolkits.mplot3d import Axes3D

# Read the data from the CSV file
data = pd.read_csv("./Datasets/cube_data.csv")

# Extract the X, Y, Distance (Z) coordinates
x = data['X'].values
y = data['Y'].values
z = data['Distance'].values

# Noise reduction using Gaussian filter
z_smoothed = gaussian_filter(z, sigma=1)

# Create grid data for mesh
grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
grid_z = griddata((x, y), z_smoothed, (grid_x, grid_y), method='cubic')

# Handle NaN values by filling them with the mean of the valid values
nan_mask = np.isnan(grid_z)
if nan_mask.any():
    grid_z[nan_mask] = np.nanmean(grid_z)

# Debug: Print min and max of grid_z before rescaling
print("Min of grid_z before rescaling:", np.nanmin(grid_z))
print("Max of grid_z before rescaling:", np.nanmax(grid_z))

# Apply contrast stretching
grid_z = exposure.rescale_intensity(grid_z, in_range=(np.nanmin(grid_z), np.nanmax(grid_z)), out_range=(0, 255))

# Debug: Print min and max of grid_z after rescaling
print("Min of grid_z after rescaling:", np.min(grid_z))
print("Max of grid_z after rescaling:", np.max(grid_z))

# Debug: Plot grid_z before and after rescaling
plt.figure()
plt.imshow(grid_z, cmap='viridis')
plt.title('Processed Data (Grid)')
plt.colorbar()
plt.show()

# Plotting the raw data
fig = plt.figure(figsize=(12, 6))

# Raw data scatter plot
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(x, y, z, c='r', marker='o')
ax1.set_title('Raw Data')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Distance')

# Mesh plot of the processed data
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(grid_x, grid_y, grid_z, cmap='viridis')
ax2.set_title('Processed Data (3D Mesh)')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Distance')

plt.show()
