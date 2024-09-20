import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from harris_corner_detector import my_corner_harris
from harris_corner_detector import my_corner_peaks

print("Corner detection in progress ... \n")

# Read the image using PIL
img = Image.open('im2.jpg')
scale_factor = 0.2
# Resize the image
img_resized = img.resize((int(img.width * scale_factor), int(img.height * scale_factor)))

# Convert the image to grayscale
img_gray = img_resized.convert('L')

# Convert image to numpy array
I_gray = np.array(img_gray)

# Normalize the grayscale image
I_gray = I_gray / 255.0

# Display the original grayscale image with red corners
plt.figure(figsize=(12, 6))

# Apply Harris corner detection to create R array
R_array = my_corner_harris(I_gray, 0.05, 2.5)

# Extract the final corners
corner_locations = my_corner_peaks(R_array, 0.00355029)

# Plot the grayscale image with red squares for corners
plt.subplot(1, 2, 1)
plt.imshow(I_gray, cmap='gray')
plt.scatter(corner_locations[:, 1], corner_locations[:, 0], color='red', marker='s', s=2)
plt.axis('off')
plt.title('Grayscale Image with Red Corners')

# Convert grayscale image back to RGB
I_rgb = np.array(img_resized.convert('RGB'))

# Overlay red corners on the RGB image using plt.scatter
plt.subplot(1, 2, 2)
plt.imshow(I_rgb)
plt.scatter(corner_locations[:, 1], corner_locations[:, 0], color='red', marker='s', s=2)
plt.axis('off')
plt.title('RGB Image with Red Corners')

plt.show()

print("The Harris Corner Detector successfully detected the desired corners !!!")
