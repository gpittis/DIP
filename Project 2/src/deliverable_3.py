import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.image import imsave
from my_img_rotation import my_img_rotation


# Load the image
RGB = np.array(Image.open('im2.jpg'))

# Resize the image
scale_factor = 0.2
RGB_resized = np.array(Image.fromarray(RGB).resize((int(RGB.shape[1] * scale_factor), int(RGB.shape[0] * scale_factor))))

print("Image Rotation_1 in progress ...\n")

# Rotate image 54*pi/180 rads
I1 = my_img_rotation(RGB_resized, 54 * np.pi / 180)
plt.figure()
plt.imshow(I1)
plt.show()
imsave('rotation_1.jpg', I1)

print("Image Rotation_1 completed successfully !!!\n")

print("Image Rotation_2 in progress ...\n")

# Rotate image 213*pi/180 rads
I2 = my_img_rotation(RGB_resized, 213 * np.pi / 180)
plt.figure()
plt.imshow(I2)
plt.show()
imsave('rotation_2.jpg', I2)

print("Image Rotation_2 completed successfully !!!")
