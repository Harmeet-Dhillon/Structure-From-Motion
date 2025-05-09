import numpy as np

def LinearPnP2(world_coords, image_coords, K):
    """
    Linear PnP to estimate the camera pose.

    :param world_coords: World coordinates.
    :type world_coords: numpy.ndarray
    :param image_coords: Image coordinates.
    :type image_coords: numpy.ndarray
    :param K: Camera matrix.
    :type K: numpy.ndarray
    :return: Camera center, Rotation matrix
    :rtype: tuple
    """

    A = []
    for i in range(len(world_coords)):
        X,Y,Z=world_coords[i]
        x,y=image_coords[i]
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -x * X, -x * Y, -x * Z, -x])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -y * X, -y * Y, -y * Z, -y])

    A = np.array(A)
    U, D, V = np.linalg.svd(A)
    P = V[-1, :].reshape(3, 4)

    R = np.linalg.inv(K) @ P[0:3, 0:3]
    Ur, Dr, Vr = np.linalg.svd(R)
    R = Ur @ Vr

    if np.linalg.det(R) < 0:
        R = -R

    #C = -np.linalg.inv(K) @ P[:, 3]
    C = -np.linalg.inv(K) @ P[:, 3]
    C /= Dr[0]

    return C,R


def LinearPnP1(Xo, ximage, K):
    n = len(Xo)  # Get the number of correspondences
    A = np.zeros((2 * n, 12))  # Initialize A with shape (2*n, 12)
    #We assume Xo is list of world points with correspondence in images..
    for i in range(n):
        X, Y, Z = Xo[i]  # Extract 3D point
        x,y = ximage[i]  # Extract 2D point
        
        # Create the two rows for the current correspondence
        r1 = np.array([[X, Y, Z, 1, 0, 0, 0, 0, -x * X, -x * Y, -x * Z, -x]])  # Shape (1, 12)
        r2 = np.array([[0, 0, 0, 0, X, Y, Z, 1, -y * X, -y * Y, -y * Z, -y]])  # Shape (1, 12)
        
        # Fill the matrix A with the new rows
        A[2 * i] = r1  # Set the row for r1
        A[2 * i + 1] = r2  # Set the row for r2

    U, S, V = np.linalg.svd(A)  # Perform Singular Value Decomposition
    # print("U",U.shape)
    # print("V",V.shape)
    # R = U @ V  # Calculate the rotation matrix
    ##note the rotation here is world w.r.t camera
    ### Getting the transpose - K^-1 * p4 / S[0] ### where p4 is the last column of P
    v = V.T  # Transpose of V
    
    ## By least squares, P is the last column on v
    P = v[:, -1]  # Last column of V

    P = P.reshape(3, 4)  # Reshape to 3x4
    ####testing
    #print("P",P)
    X_homo = Xo[0].reshape(3, 1)
    X_homo = np.vstack((X_homo, np.ones((1, X_homo.shape[1]))))
    im=P@X_homo
    im=im/im[2]
    
    K_inv = np.linalg.inv(K)
    P_=K_inv@P
    R_=P_[:,:3] # inverse of matrix K 
    #R_=K_inv@P_
    #R=R_
    U,S,V=np.linalg.svd(R_)
    R=U@V
    p4 = P_[:, -1]  # Get the last column of P
    
    C = p4  # Compute the camera center
    C = C / S[0]  # Normalize by the first singular value
    
    if np.linalg.det(R)<0:  # Check if the determinant is -1
        R = -R
        #C=-C  # Negate R if necessary
  
  # Negate C if necessary
    #print("shape of R",R.shape)
    print("all images ,u_old , v_old",ximage)
    print("R det ",np.linalg.det(R))
    # depth=S[0]
    # ###check it right there####
    def projectBack(Xp,xi,Cp,Rp):
        Cp =Cp.reshape(3,1)
        C_new=Cp

        Hp=np.hstack((Rp,C_new))
        Pp= K@Hp
        #print("Pp",Pp)
        X_homo = Xp.reshape(3, 1)
        X_homo = np.vstack((X_homo, np.ones((1, X_homo.shape[1]))))   # Convert 3D point to homogeneous coordinates
        x_reprojected = Pp @ X_homo
        u_n = x_reprojected[0] / x_reprojected[2]
        v_n = x_reprojected[1] / x_reprojected[2] 
        e = np.sqrt((xi[0]- u_n)**2 + (xi[1] - v_n)**2)
        print("new image x, y",u_n,v_n) 
        print("error is ",e)
    ###projecting world point  back
    print("world cords",Xo[0])
    for i in range(n):
        projectBack(Xo[i],ximage[i],C,R)
    #projectBack(Xo[0],np.zeros((3,1)),np.eye(3))
    # P=P.reshape(12,1)
    # Xi,Yi,Zi=Xo[0]
    # xi,yi=ximage[0]
    # ru1= np.array([[Xi, Yi, Zi, 1, 0, 0, 0, 0, -xi * Xi, -xi * Yi, -xi* Zi, -xi]])
    # ru2=np.array([[0, 0, 0, 0, Xi, Yi, Zi, 1, -yi * Xi, -yi * Yi, -yi * Zi, -yi]])
    

    # print("reverse eng",ru1,ru2)
    return C, R  # Return camera center and rotation matrix
    


