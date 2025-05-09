import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
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
    
    return np.array([q0, q1, q2,q3])



def get_params(V):
    # Count the number of observations
    ###lenght of camera_indices and point_indices list will be equal to no of observations
    num_observations = np.sum(V)

    # Lists to store camera and point indices
    camera_indices = []
    point_indices = []

    for j in range(V.shape[1]):  # Loop over world points (columns)
        for i in range(V.shape[0]):  # Loop over cameras (rows)
            if V[i,j] == 1:  # If point i is visible from camera j
                camera_indices.append(i)
                point_indices.append(j)

    return camera_indices, point_indices

def reprojectionerror(x0, K, V, image_points, n_cameras, n_points):
    """ Compute the reprojection error for optimization. """

    # Extract camera parameters from x0
    camera_params = x0[:n_cameras * 7].reshape(n_cameras, 7)  # (n_cameras, 7)
    
    # Extract world points from x0
    world_points = x0[n_cameras * 7:].reshape(n_points, 3)  # (n_points, 3)

    errors = []
    count=0
    for j in range(V.shape[1]):  # Iterate over world points
        for i in range(0, V.shape[0]):  # Iterate over cameras
            if V[i,j] == 1:  # If the point is visible in the camera

                # Extract translation (C) and quaternion (q)
                C = camera_params[i, :3]  # First 3 elements (Translation)
                q = camera_params[i, 3:]  # Last 4 elements (Quaternion)
                R=quaternion_to_rotation_matrix(q)
                # Convert quaternion to rotation matrix
                #R = quaternion_to_rotation_matrix(q)  # (3x3)

                X = world_points[j]  # World point (3D)
                x = image_points[count]  # Corresponding image point (2D)
                count+=1
                # Compute projection
                c_new = -R.T @ C.reshape(3, 1)  # Convert C to column vector
                H = np.hstack((R.T, c_new))  # (3x4) transformation matrix
                P = K @ H  # (3x4) Projection matrix

                # Convert X to homogeneous coordinates
                X_h = np.hstack([X, 1]).reshape(4, 1)  # (4,1)

                # Project 3D point to 2D
                P_X = P @ X_h  # (3,1) projected point
                px = P_X[0] / P_X[2]  # Normalize homogeneous coordinates
                py = P_X[1] / P_X[2]

                # Compute squared reprojection error
                error1 = (x[0] - px) ** 2 
                error2 = (x[1] - py) ** 2  
                errors.append(error1)
                errors.append(error2)

    return np.array(errors).flatten()



def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    num_residuals = len(camera_indices) * 2  # Two residuals per point
    num_params = n_cameras * 7 + n_points * 3  # 6 params per camera, 3 params per point

    A = lil_matrix((num_residuals, num_params), dtype=int)  # Sparse matrix to store the sparsity structure
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)
    
    i = np.arange(len(camera_indices))  # Indices for the cameras and points

    # Fill the sparsity matrix for camera parameters (6 parameters per camera: 3 translation + 3 rotation)
    for s in range(7):
        A[2 * i, camera_indices * 7 + s] = 1
        A[2 * i + 1, camera_indices * 7 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras* 7+ point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 7 + point_indices * 3 + s] = 1

    return A


def BundleAdjustment(K, V, world_points, cameraposelist, image_points):
    camera_indices, point_indices = get_params(V)
    n_cameras = len(cameraposelist)
    n_points = len(world_points)
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

    # Displaying optimization progress
    pbar = tqdm(total=100, desc="[BundleAdjustment]")

    def fun(params, K, V, image_points, n_cameras, n_points):

        error = reprojectionerror(params, K, V, image_points, n_cameras, n_points)
        reprojection_error = np.linalg.norm(error)  # Total error

        if not hasattr(fun, "initial_error"):
            fun.initial_error = reprojection_error  # Store initial error

        error_reduction = ((fun.initial_error - reprojection_error) / fun.initial_error) * 100

        pbar.update(1)
        pbar.set_postfix(error=f"{reprojection_error:.4f}", reduction=f"{error_reduction:.2f}%")

        return error
    ###breaking cameraposes , converting R into quaternion ,
    camera_list = []  # Use a list to store values before flattening

    for C_, R_ in cameraposelist:
        C_flat = C_.ravel()  # Flatten translation vector (3,)
        qconv = rotation_matrix_to_quaternion(R_)  # Convert rotation matrix to quaternion (4,)

        camera_list.extend(C_flat)  # Append translation values
        camera_list.extend(qconv)  # Append quaternion values

    cameraArray = np.array(camera_list)
    x0 = np.hstack([cameraArray, np.array(world_points).ravel()])
    res =  least_squares(fun, x0, jac_sparsity=A, x_scale='jac', ftol=1e-10,  method='trf',args=(K, V, image_points, n_cameras, n_points))
         
    pbar.close()  # Close tqdm progress bar after optimization

    optimized_cameras = res.x[:n_cameras * 7].reshape(n_cameras, 7)  # Each camera: 3 (C) + 4 (quaternion)
    optimized_world_points = res.x[n_cameras * 7:].reshape(n_points, 3)

    # Convert back from quaternion to rotation matrix
    refined_camera_poses = []
    for cam in optimized_cameras:
        C_opt = cam[:3]  # Extract translation
        R_opt = quaternion_to_rotation_matrix(cam[3:])  # Convert quaternion back to rotation matrix
        refined_camera_poses.append((C_opt, R_opt))

    return refined_camera_poses, optimized_world_points.tolist()

'''

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    num_residuals = len(camera_indices) * 2  # Two residuals per point
    num_params = n_cameras * 12 + n_points * 3  # 12 params per camera, 3 params per point

    A = lil_matrix((num_residuals, num_params), dtype=int)
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)
    
    i = np.arange(len(camera_indices))

    # Fill the sparsity matrix for camera parameters (12 parameters per camera: 3 translation + 9 rotation)
    for s in range(12):
        A[2 * i, camera_indices * 12 + s] = 1
        A[2 * i + 1, camera_indices * 12 + s] = 1

    # Fill the sparsity matrix for point parameters (3 parameters per point)
    for s in range(3):
        A[2 * i, n_cameras * 12 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 12 + point_indices * 3 + s] = 1

    return A
def BundleAdjustment(K, V, world_points, cameraposelist, image_points):
    camera_indices, point_indices = get_params(V)
    n_cameras = len(cameraposelist)
    n_points = len(world_points)
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

     # Displaying optimization progress
    pbar = tqdm(total=100, desc="[BundleAdjustment]")

    def fun(params, K, V, image_points, n_cameras, n_points):

        error = reprojectionerror(params, K, V, image_points, n_cameras, n_points)
        reprojection_error = np.linalg.norm(error)  # Total error

        if not hasattr(fun, "initial_error"):
            fun.initial_error = reprojection_error  # Store initial error

        error_reduction = ((fun.initial_error - reprojection_error) / fun.initial_error) * 100

        pbar.update(1)
        pbar.set_postfix(error=f"{reprojection_error:.4f}", reduction=f"{error_reduction:.2f}%")

        return error
    ###breaking cameraposes , converting R into quaternion ,
    camera_list = []
    for C_, R_ in cameraposelist:
        C_flat = C_.ravel()  # Flatten translation vector (3,)
        R_flat = R_.ravel()  # Flatten rotation matrix (9,)
        camera_list.extend(C_flat)
        camera_list.extend(R_flat)

    cameraArray = np.array(camera_list)
    x0 = np.hstack([cameraArray, np.array(world_points).ravel()])
    res = least_squares(fun, x0, jac_sparsity=A, x_scale='jac', ftol=1e-10, method='trf', args=(K, V, image_points, n_cameras, n_points))

    # Reshaping the result into camera parameters and world points
    optimized_cameras = res.x[:n_cameras * 12].reshape(n_cameras, 12)  # Each camera has 12 parameters
    optimized_world_points = res.x[n_cameras * 12:].reshape(n_points, 3)  # Each world point has 3 coordinates

    # Convert cameras into a list of [C, R] format
    camera_list = [[camera[:3], camera[3:].reshape(3, 3)] for camera in optimized_cameras]

    return camera_list, optimized_world_points.tolist()
'''