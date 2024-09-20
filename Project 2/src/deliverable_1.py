import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import io, color, feature, transform
from my_hough_transform import my_hough_transform

print("Hough Transform in progress ...\n")

img = io.imread('im2.jpg')
scale_factor = 0.2

# Resize the image
img_resized = transform.resize(img, (int(img.shape[0] * scale_factor), int(img.shape[1] * scale_factor)))
I1 = color.rgb2gray(img_resized)

# Adjust figure 1 to the dimensions of the image.
plt.figure(1, figsize=(img_resized.shape[1] / 100, img_resized.shape[0] / 100))
plt.imshow(img_resized)
plt.title('Original Image')
plt.axis('off')

I1 = gaussian_filter(I1, sigma=4)  # Smoothing filter. Gaussian smoothing helps to reduce the noise in the image.
binary_image = feature.canny(I1, sigma=1)  # Edge detector. I explain its function in the report.

# Initializing d_theta , d_rho and n.
N2, N1 = binary_image.shape
d_rho = 1
d_theta = 1 * np.pi / 180
n = 14
rho_max = round(np.sqrt(N1**2 + N2**2))  # max possible distance from origin
thetas = np.arange(-90, 90 + 1, d_theta * 180 / np.pi)  # range of theta values in degrees
t = np.deg2rad(thetas)  # angle in radians
rhos = np.arange(-rho_max, rho_max + 1, d_rho)  # range of rho values

H, L, res = my_hough_transform(binary_image, d_rho, d_theta, n)  # Calling my Hough Transform

# Plotting the lines
plt.figure(1)
plt.imshow(img_resized)
plt.title('Original Image')
plt.axis('off')
# Plotting the lines
x = np.arange(N1)
for k in range(L.shape[0]):
    # If the line is vertical (angle theta is 0).
    if L[k, 1] == 0:
        # Plot a vertical line at x = L[k, 0] from y = 0 to y = N2.
        plt.plot([L[k, 0], L[k, 0]], [0, N2], color='red')
    else:
        # Calculate the y-intercept for x = 0.
        y1 = (L[k, 0] - 0 * np.cos(L[k, 1])) / np.sin(L[k, 1])
        # Calculate the y-intercept for x = N1.
        y2 = (L[k, 0] - N1 * np.cos(L[k, 1])) / np.sin(L[k, 1])
        # Plot the line from (0, y1) to (N1, y2).
        plt.plot([0, N1], [y1, y2], color='red')

# Set the limits of the x-axis from 0 to N1.
plt.xlim(0, N1)
# Set the limits of the y-axis from N2 to 0 (inverted y-axis).
plt.ylim(N2, 0)
plt.show()

# Plotting Hough Transform
plt.figure(2)
plt.imshow(H, cmap='gray', extent=(float(thetas[0]), float(thetas[-1]), float(rhos[-1]), float(rhos[0])), aspect='auto', vmin=0, vmax=15)
plt.xlabel('Theta (degrees)')
plt.ylabel('Rho (pixels)')
plt.title('Hough Transform')

# Plotting peaks on Hough Transform
plt.scatter(np.rad2deg(L[:, 1]), L[:, 0], s=100, c='red', edgecolors='black')

# Plotting the binary image
plt.figure(3)
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image')
plt.axis('off')

plt.show()

print("Hough Transform completed successfully")
