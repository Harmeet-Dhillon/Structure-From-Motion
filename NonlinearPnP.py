import numpy as np
import scipy.optimize
from tqdm import tqdm

def normalize_quaternion(q):
    """ Normalize the quaternion to unit length. """
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Quaternion has zero norm, cannot normalize.")
    return q / norm

def quaternion_to_rotation_matrix(q):
    """ Converts a normalized quaternion to a 3x3 rotation matrix. """
    q = normalize_quaternion(q)  # Normalize the quaternion
    q0, q1, q2, q3 = q
    R = np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])
    return R

def rotation_matrix_to_quaternion(R):
    """ Converts a 3x3 rotation matrix to a quaternion. """
    trace = np.trace(R)
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        q0 = 0.25 * s
        q1 = (R[2, 1] - R[1, 2]) / s
        q2 = (R[0, 2] - R[2, 0]) / s
        q3 = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q0 = (R[2, 1] - R[1, 2]) / s
            q1 = 0.25 * s
            q2 = (R[0, 1] + R[1, 0]) / s
            q3 = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q0 = (R[0, 2] - R[2, 0]) / s
            q1 = (R[0, 1] + R[1, 0]) / s
            q2 = 0.25 * s
            q3 = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q0 = (R[1, 0] - R[0, 1]) / s
            q1 = (R[0, 2] + R[2, 0]) / s
            q2 = (R[1, 2] + R[2, 1]) / s
            q3 = 0.25 * s
    
    return np.array([q0, q1, q2, q3])

def NonlinearPnP(Xmatch, xmatch, K, C0, R0):
    """
    Refines the camera pose using nonlinear PnP optimization.

    Args:
        K: Camera intrinsic matrix.
        C0: Initial camera center (3x1).
        R0: Initial rotation matrix (3x3).
        Xmatch: List of 3D world points (3xN).
        xmatch: List of 2D image points (2xN).

    Returns:
        C_opt: Optimized camera center (3x1).
        R_opt: Optimized rotation matrix (3x3).
    """
    # Initial guess for the parameters: [C, quaternion(q)]
    qconv = rotation_matrix_to_quaternion(R0)  # Initial quaternion representing no rotation
    C0 = np.array(C0)
    initial_params = np.concatenate((C0.flatten(), qconv))

    # Displaying optimization progress using tqdm
    pbar = tqdm(total=100, desc="[NonLinearPnP] Optimization Progress")

    # Wrapper function around reprojection_error to track progress
    def reprojection_error_with_progress(params, K, xmatch, Xmatch):
        C = params[:3]  # Camera center
        q = params[3:7]  # Quaternion
        R = quaternion_to_rotation_matrix(q)  # Convert quaternion to rotation matrix

        # Update camera center in camera's reference frame (if required)
        C = C.reshape(3, 1)
        R = R.T
        C = -R@ C

        errors = []
        for x, X in zip(xmatch, Xmatch):
            X_h = np.hstack((X, 1)).reshape(4, 1)  # Convert to homogeneous coordinates

            # Compute the projection matrix P
            P = K @ np.hstack((R, C.reshape(3, 1)))

            # Project the 3D point
            P_X = P @ X_h
            px = P_X[0] / P_X[2]  # Normalize
            py = P_X[1] / P_X[2]
            
            # Compute the reprojection error
            #error = np.sqrt((x[0] - px) ** 2 + (x[1] - py) ** 2)
            error = (x[0] - px) ** 2 + (x[1] - py) ** 2
            errors.append(error)
            
        total_error = np.array(errors).flatten()
        
        # Calculate error reduction for progress bar
        if not hasattr(reprojection_error_with_progress, "initial_error"):
            reprojection_error_with_progress.initial_error = np.sum(total_error)

        initial_error = reprojection_error_with_progress.initial_error
        error_reduction = ((initial_error - np.sum(total_error)) / initial_error) * 100

        # Update tqdm progress bar with error and error reduction
        pbar.update(1)
        pbar.set_postfix(error=f"{np.sum(total_error):.4f}", reduction=f"{error_reduction:.2f}%")

        return total_error

    # Perform optimization using scipy's least_squares
    optimization = scipy.optimize.least_squares(
        fun=reprojection_error_with_progress,
        x0=initial_params,
        args=(K, xmatch, Xmatch),
        max_nfev=50000,
        method='lm',
        verbose=2,
        ftol=1e-12,  # tighter tolerance
        gtol=1e-12,  # gradient tolerance
        xtol=1e-12   # variable tolerance
    )
    pbar.close()

    # Extract optimized camera parameters
    optimized_params = optimization.x
    C_opt = optimized_params[:3]  # Optimized camera center
    q_opt = optimized_params[3:7]  # Optimized quaternion
    R_opt = quaternion_to_rotation_matrix(q_opt)  # Optimized rotation matrix

    return C_opt, R_opt
