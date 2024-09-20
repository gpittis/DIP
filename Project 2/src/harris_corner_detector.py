import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import convolve

# The equations referred to in the code are found in Section 2.2


def my_corner_harris(img, k, sigma):
    filter_size = round(4 * sigma)  # The length of the side of the Gaussian window is equal to round(4 * sigma).
    half_size = (filter_size - 1) / 2

    # Create a mesh grid for x and y coordinates within the Gaussian window.
    x, y = np.meshgrid(np.arange(-half_size, half_size + 1), np.arange(-half_size, half_size + 1))

    # Calculate the function g, which has non-zero values near the origin and "dies off" as (x,y) moves away from (0,0).
    # In other words, the function g resembles a "descending bell",
    # effectively attenuating the influence of distant pixels from the point (p1, p2) in the summation of equation 2.
    # Therefore, theoretically, all pixels in the image contribute to the summation,
    # but in practice, only those close to the pixel of interest are considered.
    # This significantly reduces the computational complexity of the summation.
    g = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))  # equation 1
    g = g / np.sum(g)  # Normalize the filter so that its sum equals 1.

    # The Sobel masks for computing the partial derivatives of the image in horizontal and vertical direction.
    H_horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    H_vertical = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Convolve the image with Sobel masks to calculate the partial derivatives.
    Ix = convolve2d(img, H_horizontal, mode='same')
    Iy = convolve2d(img, H_vertical, mode='same')

    # # Convert gradients to float.
    Ix = Ix.astype(float)
    Iy = Iy.astype(float)

    # I compute the elements of matrix M in equation 5.
    Ix2 = convolve(Ix * Ix, g, mode='constant', cval=0.0)
    Iy2 = convolve(Iy * Iy, g, mode='constant', cval=0.0)
    Ixy = convolve(Ix * Iy, g, mode='constant', cval=0.0)

    harris_response = (Ix2 * Iy2 - Ixy ** 2) - k * (Ix2 + Iy2) ** 2  # equation 7
    # Essentially, the matrix R acts as the detector. From its sign,
    # I can discern whether there's a corner, an edge, or a homogeneous region.

    return harris_response


def my_corner_peaks(harris_response, rel_threshold):
    rows, cols = harris_response.shape

    # Local maxima of harris_response are the points of interest in this specific case where we are looking for corners.

    # This array will be used to store the local maxima of harris_response array.
    # All values in the array are initially set to False,
    # indicating that no local maxima have been detected at any position.
    # As the code processes each pixel, it will update this array to True where local maxima are found.
    local_max = np.zeros_like(harris_response, dtype=bool)

    # Find the positions of the harris_response array at which there are local maxima.
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Extract the 3x3 neighborhood
            neighborhood = harris_response[i - 1:i + 2, j - 1:j + 2]
            if harris_response[i, j] == np.max(neighborhood):
                local_max[i, j] = True

    # Determine the threshold based on the relative threshold.
    threshold = harris_response.max() * rel_threshold

    # Keep only the corners that match the local maxima.
    harris_response = harris_response * local_max

    # Find corners that exceed the threshold.
    corners = (harris_response > threshold)

    # Get the row and column indices of the corners.
    rows, cols = np.nonzero(corners)

    corner_locations = np.column_stack((rows, cols))

    # Return the coordinates of the detected corners.
    return corner_locations

