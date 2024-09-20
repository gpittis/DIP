import numpy as np


# This function calculates and returns the equalization transform of the input image.
def get_equalization_transform_of_img(img_array):

    equalization_transform = np.zeros(256, dtype=np.uint8)  # Initialize with size 256
    levels_appearances_list = [0] * 256  # Initialize with size 256
    u_list = [0] * 256  # Initialize with size 256
    u_list_normalized = [0] * 256  # Initialize with size 256

    # Count appearances of each pixel value
    for i in img_array:
        for j in i:
            # The index (i.e., j) represents the value taken by the pixel in the input image.
            # levels_appearances_list[j] is the number of appearances of this pixel value in the input image.
            # It is an integer.
            levels_appearances_list[j] += 1

    total_pixels = img_array.shape[0] * img_array.shape[1]  # The total number of pixels in the image.


    # I use this variable to sum the number of appearances of pixel values in the input image.
    # Therefore, sum_of_levels_appearances will be an integer.
    # It is initialized to 0.
    sum_of_levels_appearances = 0

    for i in range(256):
        sum_of_levels_appearances += levels_appearances_list[i]
        # The element u_list[i] corresponds to: p(x_0) + ... + p(x_i).
        # So, u_list[i] must take values from 0 to 1,
        # where x_0 is the smallest pixel value and x_i is the i-th in sequence pixel value.
        # p(x_0) is the probability of appearance of value x_0 in the input image.
        # p(x_i) is the probability of appearance of value x_i in the input image.
        u_list[i] = sum_of_levels_appearances / total_pixels


    # Normalize u_list elements to be between 0 and 1.
    u_list_min = np.min(u_list)
    u_list_max = np.max(u_list)
    for i in range(len(u_list)):
        u_list_normalized[i] = (u_list[i] - u_list_min) / (u_list_max - u_list_min)

    total_unique_levels_number = 256   # The number of all possible pixel values range from 0 to 255.
    v0 = u_list_normalized[0]

    # I apply equation 2 and calculate the equalization transform.
    for k in range(total_unique_levels_number):
        vk = u_list_normalized[k]
        d = (vk - v0) / (1 - v0)
        num = d * (total_unique_levels_number - 1)
        # Due to round(num),some pixel values in the input image are mapped to the same pixel value in the equalized image.
        yk = round(num)
        #  k is the value of the pixel in the input image,
        #  and equalization_transform[k] is the value of this pixel in the equalized image.
        equalization_transform[k] = yk

    return equalization_transform  # It returns the equalization transform of the input image.


# This function generates the equalized image using the global equalization transform.
def perform_global_hist_equalization(img_array):

    # I calculate the equalization transform of the entire image, i.e., the global equalization transform.
    equalization_transform = get_equalization_transform_of_img(img_array)

    # I initialize the equalized image as a black image with the dimensions of the input image.
    equalized_img = np.zeros_like(img_array)
    for i in range(img_array.shape[0]):  # img_array.shape[0] = number of rows in img_array
        for j in range(img_array.shape[1]):  # img_array.shape[1] = number of columns in img_array
            # I apply the global equalization transform to the pixels of the input image to produce the equalized image.
            equalized_img[i][j] = equalization_transform[img_array[i][j]]

    return equalized_img  # It returns the equalized image.




