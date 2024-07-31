import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from skimage import exposure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay

# Read the data from the CSV file
data = pd.read_csv("./Datasets/cube_data.csv")

# Extract the X, Y, Distance (Z) coordinates
x = data['X'].values
y = data['Y'].values
z = data['Distance'].values

# Apply noise reduction using Gaussian filter
x_smoothed = gaussian_filter(x, sigma=1)
y_smoothed = gaussian_filter(y, sigma=1)
z_smoothed = gaussian_filter(z, sigma=1)

# Create a high-resolution grid for interpolation
grid_x, grid_y = np.mgrid[x.min():x.max():200j, y.min():y.max():200j]

# Interpolate the Z values on the grid
grid_z = griddata((x_smoothed, y_smoothed), z_smoothed, (grid_x, grid_y), method='cubic')

# Handle NaN values by filling with the mean of the non-NaN values
nan_mask = np.isnan(grid_z)
grid_z[nan_mask] = np.nanmean(grid_z)

# Apply contrast enhancement using contrast stretching
grid_z_stretched = exposure.rescale_intensity(grid_z, in_range=(np.nanmin(grid_z), np.nanmax(grid_z)), out_range=(0, 1))

# Create a 3D plot for raw data
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(x, y, z, c='r', marker='o')
ax1.set_title('Raw Data')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Distance')

# Create a Delaunay triangulation
points = np.column_stack((x_smoothed, y_smoothed, z_smoothed))
tri = Delaunay(points[:, :2])
vertices = points[tri.simplices]

# Create a 3D plot for the interpolated surface using Delaunay triangulation
ax2 = fig.add_subplot(122, projection='3d')
poly = Poly3DCollection(vertices, alpha=0.7, facecolor='cyan', edgecolor='grey')
ax2.add_collection3d(poly)

# Adjust the view
ax2.set_xlim([x.min(), x.max()])
ax2.set_ylim([y.min(), y.max()])
ax2.set_zlim([z.min(), z.max()])

# Label axes
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Distance')
ax2.set_title('3D Interpolated Surface of Noisy Cube with Digital Image Processing')

plt.show()
