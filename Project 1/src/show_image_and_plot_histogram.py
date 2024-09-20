import numpy as np
import matplotlib.pyplot as plt


# With these two functions, I create the histograms of the requested images.
def show_image_and_plot_histogram(img_array, name_histogram, name_photo):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img_array, cmap='gray')  # Assuming img_array is a grayscale image
    plt.title(name_photo)
    plt.axis('off')  # Hide axis lines and labels in the subplot
    # Initialize an array to store the histogram of pixel intensities
    image_hist = np.zeros(256)
    # Iterate over each pixel in the equalized image and update the histogram
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            # Increment the corresponding bin in the histogram based on the pixel intensity
            # In other words, it counts the occurrences of each pixel intensity in the image and updates the histogram accordingly.
            image_hist[img_array[i, j]] += 1
    # Calculate the total number of pixels in the image
    total_pixels = img_array.shape[0] * img_array.shape[1]

    plt.subplot(1, 2, 2)

    # It creates a bar plot with 256 bins (corresponding to the pixel values),
    # using np.arange(256) as the x-axis values and image_hist as the heights of the bars.
    # The bars are colored red with an alpha (transparency) of 0.5.
    plt.bar(np.arange(256), image_hist, color='red', alpha=0.5, width=1)
    plt.title(name_histogram)
    plt.xlabel('Pixel Value')
    plt.ylabel('Number of pixels')
    # Add vertical lines for bin edges
    for i in range(len(image_hist)):
        plt.vlines(i, 0, image_hist[i], colors='red', alpha=0.5)
    plt.tight_layout()
    plt.show()


def show_image_and_plot_histogram_for_2_images(img_array1, img_array2,  name_histogram1, name_photo1, name_histogram2, name_photo2):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(img_array1, cmap='gray')  # Assuming img_array1 is a grayscale image
    plt.title(name_photo1)
    plt.axis('off')  # Hide axis lines and labels in the subplot
    # Initialize an array to store the histogram of pixel intensities
    image_hist = np.zeros(256)
    # Iterate over each pixel in the equalized image and update the histogram
    for i in range(img_array1.shape[0]):
        for j in range(img_array1.shape[1]):
            # Increment the corresponding bin in the histogram based on the pixel intensity
            # In other words, it counts the occurrences of each pixel intensity in the image and updates the histogram accordingly.
            image_hist[img_array1[i, j]] += 1
    # Calculate the total number of pixels in the image
    total_pixels = img_array1.shape[0] * img_array1.shape[1]

    plt.subplot(2, 2, 3)

    # It creates a bar plot with 256 bins (corresponding to the pixel values),
    # using np.arange(256) as the x-axis values and image_hist as the heights of the bars.
    # The bars are colored red with an alpha (transparency) of 0.5.
    plt.bar(np.arange(256), image_hist, color='red', alpha=0.5, width=1)
    plt.title(name_histogram1)
    plt.xlabel('Pixel Value')
    plt.ylabel('Number of pixels')
    # Add vertical lines for bin edges
    for i in range(len(image_hist)):
        plt.vlines(i, 0, image_hist[i], colors='red', alpha=0.5)

    plt.subplot(2, 2, 2)
    plt.imshow(img_array2, cmap='gray')  # Assuming img_array2 is a grayscale image
    plt.title(name_photo2)
    plt.axis('off')  # Hide axis lines and labels in the subplot
    # Initialize an array to store the histogram of pixel intensities
    image_hist = np.zeros(256)
    # Iterate over each pixel in the equalized image and update the histogram
    for i in range(img_array2.shape[0]):
        for j in range(img_array2.shape[1]):
            # Increment the corresponding bin in the histogram based on the pixel intensity
            # In other words, it counts the occurrences of each pixel intensity in the image and updates the histogram accordingly.
            image_hist[img_array2[i, j]] += 1
    # Calculate the total number of pixels in the image
    total_pixels = img_array2.shape[0] * img_array2.shape[1]

    plt.subplot(2, 2, 4)

    # It creates a bar plot with 256 bins (corresponding to the pixel values),
    # using np.arange(256) as the x-axis values and image_hist as the heights of the bars.
    # The bars are colored red with an alpha (transparency) of 0.5.
    plt.bar(np.arange(256), image_hist, color='red', alpha=0.5, width=1)
    plt.title(name_histogram2)
    plt.xlabel('Pixel Value')
    plt.ylabel('Number of pixels')
    # Add vertical lines for bin edges
    for i in range(len(image_hist)):
        plt.vlines(i, 0, image_hist[i], colors='red', alpha=0.5)

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.tight_layout()
    plt.show()

