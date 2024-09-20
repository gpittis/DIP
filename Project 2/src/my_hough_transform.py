import numpy as np
from scipy.ndimage import maximum_filter


# img_binary : img_binary is the binary image. All its pixels are black except for those that correspond to edges,
# which are white.

def my_hough_transform(img_binary, d_rho, d_theta, n):
    N2, N1 = img_binary.shape  # Get the dimensions of the binary image

    # The variable res represents the number of points in the input image that do not belong to the n detected lines.
    # I initialize the variable res to be equal to the total number of pixels in the image.
    # Then, I will find the n lines in the image and count the number of pixels that lie on these lines.
    # Finally, I will subtract the number of pixels on the detected lines from the total pixels in the image,
    # and thus find the final value of res.
    res = N1 * N2

    rho_max = round(np.sqrt(N1**2 + N2**2))  # Max possible distance from origin

    thetas = np.arange(-90, 90 + 1, d_theta * 180 / np.pi)  # Range of theta (in degrees)

    t_rad = np.deg2rad(thetas)  # I convert the values of thetas array from degrees to radians

    rhos = np.arange(-rho_max, rho_max + 1, d_rho)  # Range of rhos values

    # I initialize the RxT matrix H , where R is the length of the rhos array and T is the length of the thetas array.
    # Each cell of the matrix H, initially contains 0 votes.
    # Each pair (rhos[i],thetas[i]) represents a line. Each cell of the matrix H represents a line.
    #  Each pixel of the image will vote the cells of the H array representing the lines it is on.
    H = np.zeros((len(rhos), len(thetas)))

    # Apply the Hough Transform
    for n1 in range(N1):
        for n2 in range(N2):
            if img_binary[n2, n1] != 0:  # If the pixel is white, meaning if it represents an edge.
                for j in range(len(thetas)):
                    r = n1 * np.cos(t_rad[j]) + n2 * np.sin(t_rad[j])
                    i = np.argmin(np.abs(rhos - r))  # Find the value closest to this.
                    H[i, j] += 1  # Filling the H array with votes.

    # Local Maxima : The cells of the H array that will gather the most votes represent the local maxima.

    # Create a binary mask where the value is True,
    # if the corresponding element in H array is equal to the maximum value in its 3x3 neighborhood,
    # and False otherwise.
    local_max_binary = (H == maximum_filter(H, size=(3, 3)))

    local_max = local_max_binary * H  # This retains only those elements of H that correspond to local maxima,
    # while setting all other elements to zero. The local_max array contains the local maxima.

    max_cells = []  # Initialize a list to store the indices of the most voted cells.
    count = 0  # count is the number of pixels that lie on the lines corresponding to the most voted cells.

    # Find the n highest values and their indices. In other words, I want to find the n strongest lines.
    for z in range(n):
        max_value = -1
        max_index = (-1, -1)

        # Find the maximum value and its index.
        for i in range(local_max.shape[0]):
            for j in range(local_max.shape[1]):
                if local_max[i, j] > max_value:
                    max_value = local_max[i, j]
                    max_index = (i, j)

        if max_value <= 0:
            break

        count += max_value
        max_cells.append(max_index)
        local_max[max_index[0], max_index[1]] = 0  # Remove the found max cell.

    # Construct the L matrix with the parameters rho and theta of the n strongest lines.
    L = []
    for row, col in max_cells:
        rho = rhos[row]
        theta = t_rad[col]
        L.append([rho, theta])

    L = np.array(L)

    # Calculate the remaining pixels not belonging to edges.
    res -= count

    return H, L, res

