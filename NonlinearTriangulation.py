import numpy as np
import scipy.optimize
from tqdm import tqdm

from LinearTriangulation import LinearTriangulation


def projectionError(Xo, K, c, R, matching_list):   
    #### here we define the error in projection
    # Here Rotation R is of camera w.r.t to world
    # So world w.r.t camera becomes  R.T and -R.T@C

    X0_reshaped = Xo.reshape(-1, 3)
    X0_homogeneous = np.hstack((X0_reshaped, np.ones((X0_reshaped.shape[0], 1))))

    errors = []  # Store individual errors as 3×1 column vectors
    for match, X in zip(matching_list, X0_homogeneous): 
        _, x = match
        c_new = -R.T@ c  # getting last column
        H = np.hstack((R.T, c_new.reshape(-1, 1)))  
        #print("X_homogenous",X)
        # Matrix of shape 3x4
        P = K @ H  # Projection matrix (3x4)

        X_h = X.reshape(4, 1)  # Ensure X is in homogeneous form
        P_X = P @ X_h  # Multiply 3x4 with 4x1 column vector
        #print("X_homogenous 2",X_h)
        px = P_X[0] / P_X[2]  # Normalize homogeneous coordinates
        py = P_X[1] / P_X[2]

        # Compute reprojection error as a 3×1 column vector
        # error = np.array([[(x[0] - px) ** 2 + (x[1] - py) ** 2], [0], [0]]) 
        error = np.array([float((x[0] - px) ** 2 + (x[1] - py) ** 2), 0, 0]) 

        errors.append(error)

    return np.vstack(errors).flatten()  # Return as a flattened array for optimization

def NonlinearTriangulation(K, c, R, X0,matching_list):  
    # matching_list is p1, p2 in this 
    #X0, selected_indices = LinearTriangulation(np.zeros((3,1)), np.eye(3), c, R, K, matching_list)  # Initial triangulation
    #matching_list_filtered = [matching_list[i] for i in indexlist]
    X0_array = np.array(X0, dtype=float).ravel()

    # Displaying optimization progress
    pbar = tqdm(total=100, desc="[NonLinearTraingulation]")  # Set an appropriate total if known

    # wrapper around projectionError to track optimization progress
    def projectionError_with_progress(X, K, c, R, matching_list):
        error = projectionError(X, K, c, R, matching_list)
        reprojection_error = np.linalg.norm(error)  # Total error

        if not hasattr(projectionError_with_progress, "initial_error"):
            projectionError_with_progress.initial_error = None  # Define static variable

        if projectionError_with_progress.initial_error is None:
            projectionError_with_progress.initial_error = reprojection_error  # Store first error
        
        error_reduction = ((projectionError_with_progress.initial_error - reprojection_error) /
                       projectionError_with_progress.initial_error) * 100

        pbar.update(1)
        pbar.set_postfix(error=f"{reprojection_error:.4f}", reduction=f"{error_reduction:.2f}%")  

        return error

    optimization = scipy.optimize.least_squares(fun=projectionError_with_progress, x0=X0_array, method="lm", args=(K, c, R, matching_list), verbose=2,max_nfev=40000)
    return optimization.x.reshape(-1, 3) # Return list of 3x1 column vectors
