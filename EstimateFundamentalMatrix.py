'''
import numpy as np


def EstimateFundamentalMatrix(matches):
    """
    Computes the fundamental matrix from matched feature points.

    Args:
        matches (list(dict)): List of matches specified in the following dictionary format:
            - 'num_matches' (int): Number of matches for this location in first image
            - 'color' (tuple of float): RGB color values (r, g, b).
            - 'current_image_coordinates' (tuple of float): Coordinates (u_image_1, v_image_1) in the first image.
            - 'matching_image_coordinates' (tuple of float): Corresponding coordinates (u_image_2, v_image_2) in the second image.

    Returns:
        (Depends on implementation) The computed fundamental matrix.
    """

    assert len(matches)>= 8, "At least 8 matches are required to compute the fundamental matrix."

    num_matches = len(matches)
    A = np.zeros((num_matches, 9))

    x1 = np.ones((3, 1)) # for testing only
    x2 = np.ones((1, 3)) # for testing only

    for i, match in enumerate(matches):
        x, y = match["current_image_coordinates"]
        x_p, y_p = match["matching_image_coordinates"]
        
        if i == 0: # for testing only
            x1[0] = x
            x1[1] = y

            x2[0][0] = x_p
            x2[0][1] = y_p 

        A[i] = [x * x_p, x * y_p, x, y * x_p, y * y_p, y, x_p, y_p, 1]

    # Compute SVD
    U, S, Vt = np.linalg.svd(A)  

    # Extract the last column of V (last row of Vt since Vt is transposed)
    x = Vt[-1, :]  

    # Reshape x into a 3x3 matrix and transpose
    F = x.reshape(3, 3).T 

    # Compute SVD of F
    U, S, Vt = np.linalg.svd(F)

    # Set the smallest singular value to zero
    S[-1] = 0  

    # Reconstruct the matrix with the modified singular values
    F_rank2 = U @ np.diag(S) @ Vt

    # print(np.linalg.matrix_rank(F_rank2))

    # Normalize F_rank2 by its last element (bottom-right entry) for more stable numerical computations
    F_rank2 /= F_rank2[-1, -1]

    return F_rank2
'''

import numpy as np

   
def EstimateFundamentalMatrix(matches):
    """
    Computes the fundamental matrix from matched feature points.

    Args:
        matches (list(dict)): List of matches specified in the following dictionary format:
            - 'num_matches' (int): Number of matches for this location in first image
            - 'color' (tuple of float): RGB color values (r, g, b).
            - 'current_image_coordinates' (tuple of float): Coordinates (u_image_1, v_image_1) in the first image.
            - 'matching_image_coordinates' (tuple of float): Corresponding coordinates (u_image_2, v_image_2) in the second image.

    Returns:
        (Depends on implementation) The computed fundamental matrix.
    """

    assert len(matches)>= 8, "At least 8 matches are required to compute the fundamental matrix."

    num_matches = len(matches)
    A = np.zeros((num_matches, 9))

    x1 = np.ones((3, 1)) # for testing only
    x2 = np.ones((1, 3)) # for testing only
   
    for i, match in enumerate(matches):
        x, y = match["current_image_coordinates"]
        x_p, y_p = match["matching_image_coordinates"]
        
        if i == 0: # for testing only
            x1[0] = x
            x1[1] = y

            x2[0][0] = x_p
            x2[0][1] = y_p 

        A[i] = [x * x_p, x * y_p, x, y * x_p, y * y_p, y, x_p, y_p, 1]

    # Compute SVD
    U, S, Vt = np.linalg.svd(A)  

    # Extract the last column of V (last row of Vt since Vt is transposed)
    x = Vt[-1, :]  
    #print("x",x)
    x=x.T
    # Reshape x into a 3x3 matrix and transpose
    F = x.reshape(3, 3)

    # Compute SVD of F
    U, S, Vt = np.linalg.svd(F)

    # Set the smallest singular value to zero
    S[-1] = 0  

    # Reconstruct the matrix with the modified singular values
    F_rank2 = U @ np.diag(S) @ Vt

    #print(np.linalg.matrix_rank(F_rank2))

    # Normalize F_rank2 by its last element (bottom-right entry) for more stable numerical computations
    F_rank2 /= F_rank2[-1, -1]

    return F_rank2
