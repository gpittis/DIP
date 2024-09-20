import numpy as np


def my_img_rotation(img, angle):
    dims = img.shape  # Take the dimensions of the input image.

    # The function should work independently of the number of channels in the input image.
    if len(dims) == 2:
        # The grayscale image has 2 dimensions. Therefore, I add an extra dimension,
        # in which there will be the value of every pixel of the input image.
        img = img[:, :, np.newaxis]  # So its shape will be (img.shape[0], img.shape[1], 1).

    # The RGB image has 3 dimensions and in the third dimension,
    # it contains a 1x3 color vector for each pixel and has the shape (img.shape[0], img.shape[1], 3).

    # Convert image to double
    img = img.astype(float)

    # Define the rotation matrix based on the given angle
    t_rot = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]])

    # Calculate the center of the image
    center = np.array([dims[0] / 2, dims[1] / 2])

    # Define the translation matrix to move the image center to the origin.
    t_center = np.array([[1, 0, -center[0]],
                         [0, 1, -center[1]],
                         [0, 0, 1]])

    # This array keeps track of the x-coordinates of all pixels after rotation.
    center_rot_x = np.zeros((dims[0], dims[1]))

    # This array keeps track of the y-coordinates of all pixels after rotation.
    center_rot_y = np.zeros((dims[0], dims[1]))

    for i in range(dims[0]):
        for j in range(dims[1]):
            # Create a homogeneous coordinate for the current pixel (i, j)
            # and apply the translation matrix to move the center of the image to the origin,
            # followed by the rotation matrix to rotate the pixel around the origin.
            input_pixel = np.array([i, j, 1])
            center_rotated_pixels = np.dot(t_rot, np.dot(t_center, input_pixel.T))
            center_rot_x[i, j] = center_rotated_pixels[0]
            center_rot_y[i, j] = center_rotated_pixels[1]

    # Find the min and max coordinates of the rotated image to determine the new dimensions.
    # The output image rot_img must have the appropriate dimensions to accommodate the entire rotated input image.
    min_x = np.min(center_rot_x)
    min_y = np.min(center_rot_y)
    max_x = np.max(center_rot_x)
    max_y = np.max(center_rot_y)

    # Create an output image with appropriate size and initialize to zero
    rot_img = np.zeros((int(max_x - min_x) + 1, int(max_y - min_y) + 1, img.shape[2]))

    for i in range(rot_img.shape[0]):
        for j in range(rot_img.shape[1]):
            # Calculate the corresponding coordinates in the original image
            # Transform (i, j) from the center of the output image.
            x = (i - (rot_img.shape[0] // 2)) * np.cos(angle) + (j - (rot_img.shape[1] // 2)) * np.sin(angle)
            y = -(i - (rot_img.shape[0] // 2)) * np.sin(angle) + (j - (rot_img.shape[1] // 2)) * np.cos(angle)

            # Translate the coordinates back to the original image's coordinate system
            x = int(round(x) + center[0])
            y = int(round(y) + center[1])

            # Bilinear interpolation
            if 1 <= x < dims[0] - 1 and 1 <= y < dims[1] - 1:
                rot_img[i, j, :] = (img[x - 1, y, :] + img[x + 1, y, :] + img[x, y - 1, :] + img[x, y + 1, :]) / 4

    return rot_img.astype(np.uint8)
