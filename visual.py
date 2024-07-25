import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN

# Read the data from the CSV file
data = pd.read_csv("ultrasonic_data.csv")

# Extract the X, Y, Distance (Z) coordinates
x = data['X'].values
y = data['Y'].values
z = data['Distance'].values

# Apply noise reduction using Gaussian filter
# Smoothing the data to reduce the effect of noise
x_smoothed = gaussian_filter(x, sigma=1)
y_smoothed = gaussian_filter(y, sigma=1)
z_smoothed = gaussian_filter(z, sigma=1)

# Use clustering for segmentation (DBSCAN for example)
# Fit DBSCAN to identify clusters in the data
coords = np.vstack((x_smoothed, y_smoothed, z_smoothed)).T
db = DBSCAN(eps=1.0, min_samples=10).fit(coords)
labels = db.labels_

# Create a color map based on clusters
unique_labels = np.unique(labels)
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each cluster with a different color
for k in unique_labels:
    class_member_mask = (labels == k)
    ax.scatter(x[class_member_mask], y[class_member_mask], z[class_member_mask], 
               color=colors[k], label=f'Cluster {k}', s=5)  # Adjust the size of the points

# Label axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Distance')
ax.set_title('3D Representation of a Noisy Cube with Digital Image Processing')

# Show legend
ax.legend()

plt.show()
