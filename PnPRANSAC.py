import numpy as np
import random
from LinearPnP import LinearPnP1,LinearPnP2  # Import your LinearPnP function

def ransac_pose_estimation(X, x, K):
    """
    Estimates the camera pose using RANSAC, using your LinearPnP implementation.

    Args:
        X: 3D world points (Nx3 numpy array).
        x: 2D image points (Nx2 numpy array).
        K: Camera intrinsic matrix (3x3 numpy array).
        M: Number of RANSAC iterations.
        N: Number of total correspondences.
        epsilon_r: Error threshold for inliers.

    Returns:
        Sin: The set of inlier indices that support the best pose.
    """
    
    n = 0  # Initialize the largest inlier set size
    Sin = set()  # Initialize the best inlier set
    M = 1000
    N = len(X)
    epsilon_r = 25
    X_inliers=[]
    x_inliers=[]
    indices_list=[]
    best_C=np.zeros((3, 1))
    best_R=np.zeros((3, 3))
    for i in range(M):
        # 1. Randomly select 6 correspondences
        indices = random.sample(range(N),6)
        #indices=list(range(6))
        X_sample =[X[i] for i in indices]

        x_sample = [x[i] for i in indices]

        # 2. Estimate the camera pose [C|R] using LinearPnP
        C1, R1 = LinearPnP1(X_sample, x_sample, K)
        C2, R2 = LinearPnP2(X_sample, x_sample, K)

        # Construct the projection matrix P

        C =C1.reshape(3,1)
        R=R1
        C_new=C
        H=np.hstack((R,C_new))
        P= K@H
        
        S = []  # Initialize the current inlier set
        
        # 3. Iterate through all correspondences
        for j in range(N):
            # 4. Calculate the re-projection error
            X_homo = X[j].reshape(3, 1)
            X_homo = np.vstack((X_homo, np.ones((1, X_homo.shape[1]))))   # Convert 3D point to homogeneous coordinates
            x_reprojected = P @ X_homo

            # Normalize the reprojected coordinates
            u_reprojected = x_reprojected[0] / x_reprojected[2]
            v_reprojected = x_reprojected[1] / x_reprojected[2]

            u = x[j][0]
            v = x[j][1]

            e = np.sqrt((u - u_reprojected)**2 + (v - v_reprojected)**2)
            
            # 5. Check if the error is below the threshold
            if e < epsilon_r:
                print("error is ",e)
                print("j is",j)
                S.append(j)  # Add the index to the inlier set

        # 6. Update the best inlier set if necessary
        if n < len(S):
            n = len(S)
            X_inliers=[]
            x_inliers=[]
            indices_list=[]
            best_C=C
            best_R=R
            X_inliers = [X[i] for i in S]
            x_inliers = [x[i] for i in S]
            indices_list=[i for i in S]
    #now take the best C and R here and convert them  such in camera w.r.t world
    best_C,best_R = LinearPnP1(X_inliers, x_inliers, K)
    best_R=best_R.T
    best_C=-best_R @ best_C
    return best_C,best_R,X_inliers,x_inliers,indices_list
