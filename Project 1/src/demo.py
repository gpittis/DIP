import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from show_image_and_plot_histogram import show_image_and_plot_histogram
from adaptive_hist_eq import calculate_eq_transformations_of_regions
from adaptive_hist_eq import perform_adaptive_hist_equalization
from global_hist_eq import perform_global_hist_equalization
from show_image_and_plot_histogram import show_image_and_plot_histogram_for_2_images
from global_hist_eq import get_equalization_transform_of_img

# In the demo, I combine all the functions I created to generate the requested images and their corresponding histograms.

filename = "C:\\python\\lesson04\\input_img.png"
img = Image.open(fp=filename)
# Keep only the Luminance component of the image
bw_img = img.convert("L")
# Obtain the underlying np array
img_array = np.array(bw_img, dtype=np.uint8)

equalization_transform_input_img = get_equalization_transform_of_img(img_array)
plt.figure(figsize=(8, 6))
plt.plot(equalization_transform_input_img, color='blue')
plt.title('Equalization Transform of Input Image')
plt.xlabel('Input Pixel Value')
plt.ylabel('Transformed Pixel Value')
plt.grid(True)
plt.show()

equalized_img_global = perform_global_hist_equalization(img_array)
equalized_img_adaptive = perform_adaptive_hist_equalization(img_array, 64, 48)

show_image_and_plot_histogram(img_array, "Input Image Histogram", "Input Image")
show_image_and_plot_histogram_for_2_images(img_array, equalized_img_global, "Input Image Histogram", "Input Image", "Equalized image Histogram (Global Equalization)", "Equalized Image")
show_image_and_plot_histogram_for_2_images(img_array, equalized_img_adaptive, "Input Image Histogram", "Input Image", "Equalized image Histogram (Adaptive Equalization)", "Equalized Image")

region_len_w = 48
region_len_h = 64
# I initialize the equalized image as a black image with the dimensions of the input image.
equalized_img_wrong_adaptive = np.zeros_like(img_array)

regions_dictionary = calculate_eq_transformations_of_regions(img_array, region_len_h, region_len_w)

# The dictionary (centers_dictionary) where the keys are tuples representing contextual regions
# and the values are the centers of the corresponding contextual regions.
centers_dictionary = {}

for key in regions_dictionary:
    # I calculate the center of every contextual region.
    centers_dictionary[key] = [key[0] + (region_len_w - 1) // 2, key[1] + (region_len_h - 1) // 2]

# In centers_transform_dictionary, each key represents a contextual region center,
# and each value represents the equalization transform of the corresponding center.
centers_transform_dictionary = {}
for region in regions_dictionary:
    centers_transform_dictionary[tuple(centers_dictionary[region])] = regions_dictionary[region]

# If a pixel of the input image corresponds to the center of a contextual region,
# then I map it to the equalized image using the equalization transform of the contextual region centered at that specific pixel.
for key in centers_transform_dictionary:
    if key[0] < img_array.shape[0] and key[1] < img_array.shape[1]:
        equalized_img_wrong_adaptive[key[0]][key[1]] = centers_transform_dictionary[key][img_array[key[0]][key[1]]]

for key in regions_dictionary:
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            if (i >= key[0]) and i <= (key[0] + region_len_w - 1) and (j >= key[1]) and (j <= (key[1] + region_len_h - 1)):
                # Each pixel is mapped to the final image using the equalization transform of the region it belongs to.
                equalized_img_wrong_adaptive[i][j] = regions_dictionary[key][img_array[i][j]]

show_image_and_plot_histogram(equalized_img_wrong_adaptive, "Equalized image Histogram (Adaptive Wrong Equalization)", "Equalized Image")

output_filename_global = "C:\\python\\lesson04\\equalized_img_global.png"
equalized_img_global = Image.fromarray(equalized_img_global)
equalized_img_global.save(output_filename_global)

output_filename_adaptive = "C:\\python\\lesson04\\equalized_img_adaptive.png"
equalized_img_adaptive = Image.fromarray(equalized_img_adaptive)
equalized_img_adaptive.save(output_filename_adaptive)

output_filename_wrong_adaptive = "C:\\python\\lesson04\\equalized_img_wrong_adaptive.png"
equalized_img_wrong_adaptive = Image.fromarray(equalized_img_wrong_adaptive)
equalized_img_wrong_adaptive.save(output_filename_wrong_adaptive)

output_filename_input_grayscale = "C:\\python\\lesson04\\input_img_grayscale.png"
img_array = Image.fromarray(img_array)
img_array.save(output_filename_input_grayscale)
