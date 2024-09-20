import numpy as np
from global_hist_eq import get_equalization_transform_of_img


# This function takes as input an image.
# It also takes as input the height and width of the contextual regions into which I will divide the input image.
# It returns the equalization transform for each contextual region (in a dictionary).
def calculate_eq_transformations_of_regions(img_array: np.ndarray, region_len_h: int, region_len_w: int):
    # In this dictionary, each key is a tuple representing a specific contextual region.
    # The region_to_eq_transform[key] is the equalization transform of the corresponding contextual region.
    region_to_eq_transform = {}

    img_w = img_array.shape[0]
    img_h = img_array.shape[1]
    for i in range(0, img_w, region_len_w):
        for j in range(0, img_h, region_len_h):
            # I divide the input image into individual regions (contextual regions).
            # Each such region is a sub-image of the input image.
            # For each such sub-image, I calculate its equalization transform.
            eq_transform = get_equalization_transform_of_img(img_array[i:i + region_len_w, j:j + region_len_h])

            # I write (i, j) because in indexing, the first coordinate is the width and the second coordinate is the height.
            # Otherwise, if I wrote (j, i), I would have an error.
            region_to_eq_transform[(i, j)] = eq_transform

    return region_to_eq_transform  # It returns a dictionary


# This function takes as input an image.
# It also takes as input the height and width of the contextual regions of the input image.
# It returns the equalized image, which is generated using adaptive equalization transform.
def perform_adaptive_hist_equalization(img_array, region_len_h, region_len_w):

    # I initialize the equalized image as a black image with the dimensions of the input image.
    equalized_img = np.zeros_like(img_array)

    # The dictionary (centers_dict) where the keys are tuples representing contextual regions
    # and the values are the centers of the corresponding contextual regions.
    centers_dict = {}

    region_to_eq_transform = calculate_eq_transformations_of_regions(img_array, region_len_h, region_len_w)
    for key in region_to_eq_transform:
        # I calculate the center of every contextual region.
        centers_dict[key] = [key[0] + (region_len_w - 1)//2, key[1] + (region_len_h - 1)//2]

    # Finds the smallest and largest x and y coordinates of the tuples of contextual regions.
    # These 4 coordinates are needed to determine which contextual regions represent the 4 corners of the input image.
    max_x_region = max(region_to_eq_transform)[0]
    min_x_region = min(region_to_eq_transform)[0]
    max_y_region = max(region_to_eq_transform)[1]
    min_y_region = min(region_to_eq_transform)[1]

    left_down_corner = (min_x_region, min_y_region)   # The left down corner of the input image.
    left_up_corner = (min_x_region, max_y_region)     # The left up corner of the input image.
    right_down_corner = (max_x_region, min_y_region)  # The right down corner of the input image.
    right_up_corner = (max_x_region, max_y_region)    # The right up corner of the input image.

    # This list contains the tuples corresponding to the 4 corners of the image.
    corners = [left_down_corner, left_up_corner, right_down_corner, right_up_corner]

    # I initialize the lists that will contain the tuples representing the boundary contextual regions of the input image.
    # left_border : contains the tuples that represent the contextual regions of the left border of the input image.
    # right_border : contains the tuples that represent the contextual regions of the right border of the input image.
    # up_border : contains the tuples that represent the contextual regions of the upper border of the input image.
    # down_border : contains the tuples that represent the contextual regions of the down border of the input image.
    left_border, right_border, up_border, down_border = [], [], [], []

    # Filling the lists with the correct tuples.
    for region in region_to_eq_transform:
        if region[0] == min_x_region and region not in corners:
            left_border.append(region)
        elif region[0] == max_x_region and region not in corners:
            right_border.append(region)
        elif region[1] == max_y_region and region not in corners:
            up_border.append(region)
        elif region[1] == min_y_region and region not in corners:
            down_border.append(region)

    """
    I find which pixels of the input image are in the red area (Figure 7) of the left-down-corner. 
    In other words, I locate the outer points that exist in the left-down-corner of the input image. 
    To find the value of these pixels in the equalized image, 
    I then apply the equalization transform of the contextual region representing that specific corner 
    to the corresponding pixels of the input image. I apply the same method for the remaining corners.
    """
    region = left_down_corner
    region_center = centers_dict[region]
    for i in range(region[0], region[0] + region_len_w):
        for j in range(region[1], region[1] + region_len_h):
            if i < region_center[0] or (i >= region_center[0] and j < region_center[1]):
                equalized_img[i][j] = region_to_eq_transform[region][img_array[i][j]]

    # I apply the same method described above for the left up corner.
    region = left_up_corner
    region_center = centers_dict[region]
    for i in range(region[0], region[0] + region_len_w):
        for j in range(region[1], region[1] + region_len_h):
            if i < region_center[0] or (i >= region_center[0] and j > region_center[1]):
                if j < img_array.shape[1]:
                    equalized_img[i][j] = region_to_eq_transform[region][img_array[i][j]]

    # I apply the same method described above for the right down corner.
    region = right_down_corner
    region_center = centers_dict[region]
    for i in range(region[0], region[0] + region_len_w):
        for j in range(region[1], region[1] + region_len_h):
            if i > region_center[0] or (i <= region_center[0] and j < region_center[1]):
                if i < img_array.shape[0]:
                    equalized_img[i][j] = region_to_eq_transform[region][img_array[i][j]]

    # I apply the same method described above for the right up corner.
    region = right_up_corner
    region_center = centers_dict[region]
    for i in range(region[0], region[0] + region_len_w):
        for j in range(region[1], region[1] + region_len_h):
            if i > region_center[0] or (i <= region_center[0] and j > region_center[1]):
                if j < img_array.shape[1] and i < img_array.shape[0]:
                    equalized_img[i][j] = region_to_eq_transform[region][img_array[i][j]]

    """
    For a contextual region located on the left border of the image, I find the outer points that exist in it. 
    To find the value of these pixels in the equalized image, 
    I apply the equalization transform of the boundary contextual region ,to which the pixel is located,
    on the corresponding pixels of the input image.
    """
    for region in left_border:
        region_center = centers_dict[region]
        for i in range(region[0], region[0] + region_len_w):
            for j in range(region[1], region[1] + region_len_h):
                if i < region_center[0]:
                    equalized_img[i][j] = region_to_eq_transform[region][img_array[i][j]]

    # I apply the same method as the left border for the right border.
    for region in right_border:
        region_center = centers_dict[region]
        for i in range(region[0], region[0] + region_len_w):
            for j in range(region[1], region[1] + region_len_h):
                if i > region_center[0] and (i < img_array.shape[0]):
                    equalized_img[i][j] = region_to_eq_transform[region][img_array[i][j]]

    # I apply the same method as the left border for the up border.
    for region in up_border:
        region_center = centers_dict[region]
        for i in range(region[0], region[0] + region_len_w):
            for j in range(region[1], region[1] + region_len_h):
                if j > region_center[1] and (j < img_array.shape[1]) and i < img_array.shape[0]:
                    equalized_img[i][j] = region_to_eq_transform[region][img_array[i][j]]

    # I apply the same method as the left border for the down border.
    for region in down_border:
        region_center = centers_dict[region]
        for i in range(region[0], region[0] + region_len_w):
            for j in range(region[1], region[1] + region_len_h):
                if j < region_center[1] and i < img_array.shape[0]:
                    equalized_img[i][j] = region_to_eq_transform[region][img_array[i][j]]

    # In dict_coord, each key represents a contextual region center,
    # and each value represents the equalization transform of the corresponding center.
    dict_coord = {}
    for region in region_to_eq_transform:
        dict_coord[tuple(centers_dict[region])] = region_to_eq_transform[region]

    # If a pixel of the input image corresponds to the center of a contextual region,
    # then I map it to the equalized image using the equalization transform of the contextual region centered at that specific pixel.
    for key in dict_coord:
        if key[0] < img_array.shape[0] and key[1] < img_array.shape[1]:
            equalized_img[key[0]][key[1]] = dict_coord[key][img_array[key[0]][key[1]]]

    for key in dict_coord:  # For a center of a contextual region.
        # I check if the specific center can have neighboring centers.
        # In other words, I check if the potential neighboring centers are within the dimensions of the image.
        if (key[0] + region_len_w < img_array.shape[0]) and (key[1] - (region_len_h//2 - 1) > 0):
            # If the specific center has neighboring centers.
            # Then I create a new region.
            # The specific center and its 3 neighbors will form the vertices of this new region.
            first_c = key # Then this specific center is the first center --> [ (h-,w-) in figure 4 ].
            second_c = (key[0], key[1] - region_len_h)  # calculate the coordinates of second center --> [ (h+,w-) in figure 4 ].
            third_c = (key[0] + region_len_w, key[1])  # # calculate the coordinates of third center --> [ (h-,w+) in figure 4 ].
            fourth_c = (key[0] + region_len_w, key[1] - region_len_h)  # calculate the coordinates of fourth center --> [ (h+,w+) in figure 4 ].
            for i in range(img_array.shape[0]):
                for j in range(img_array.shape[1]):
                    # If the pixel in the input image does not coincide with the center of any contextual region
                    # (the representation of the centers of the contextual regions in the equalized image was calculated earlier).
                    if (i, j) not in dict_coord:
                        # If the pixel exists within the area defined by the 4 centers.
                        if (i >= key[0]) and (i <= (key[0] + region_len_w)) and (j >= (key[1] - region_len_h)) and (j <= key[1]):
                            # Then I apply the bilinear interpolation described in equation 4.
                            a = (i - key[0]) / ((key[0] + region_len_w) - key[0])
                            b = (j - key[1]) / ((key[1] - region_len_h) - key[1])
                            t1 = (1 - a) * (1 - b) * dict_coord[first_c][img_array[i][j]]
                            t2 = (1 - a) * b * dict_coord[second_c][img_array[i][j]]
                            t3 = a * (1 - b) * dict_coord[third_c][img_array[i][j]]
                            t4 = a * b * dict_coord[fourth_c][img_array[i][j]]
                            equalized_img[i][j] = t1 + t2 + t3 + t4

    return equalized_img  # Ιt returns the equalized image



