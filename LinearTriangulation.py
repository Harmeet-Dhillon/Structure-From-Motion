import numpy as np

def LinearTriangulation(c1,R1,c2,R2,K,matching_list):
        world_points = []
        selected_indices = []
        for index, (point1, point2) in enumerate(matching_list):  # Assuming list of tuples (p1, p2)
            col1, row1 = point1  # Extract row and column

            # Convert to homogeneous coordinates
            x_im = np.array([[col1], [row1], [1]])  # 3×1 vector

            # Camera matrix
            R1_inv = R1.T # Assuming we use R1; modify if needed
            C = c1.reshape((3, 1))  # Translation as column vector
            
            C_new=-R1_inv @ C
            H_im=np.hstack((R1_inv,C_new)) # stacking column wise ...here R is world w.r.t to camera and C is translation of 
                                    ####camera w.r.t to world

            # Compute projection matrix
            P1 = K @ H_im  # Shape (3×4)

            # Compute cross product of x_im with each column of P
            A_im = np.zeros((3, 4))  # Resultant matrix of cross products
            for i in range(P1.shape[1]):
                A_im[:, i] = np.cross(x_im.flatten(), P1[:, i])

            col2, row2 = point2  # Extract row and column

            # Convert to homogeneous coordinates
            x_cm = np.array([[col2], [row2], [1]])  # 3×1 vector

            # Camera matrix
            # Camera matrix
            R2_inv = R2.T# Assuming we use R1; modify if needed
            C = c2.reshape((3, 1))  # Translation as column vector
            
            C_new=-R2_inv @ C
            H_cm=np.hstack((R2_inv,C_new)) # stacking column wise ...here R is world w.r.t to camera and C is translation of 
                                    ####camera w.r.t to world

            # Compute projection matrix
            P2 = K @ H_cm  # Shape (3×4)

            # Compute cross product of x_im with each column of P
            A_cm = np.zeros((3, 4))  # Resultant matrix of cross products
            for i in range(P2.shape[1]):
                A_cm[:, i] = np.cross(x_cm.flatten(), P2[:, i])
         
            A = np.vstack((A_im, A_cm))
            u,s,v=np.linalg.svd(A)
            V=v.T
            V=V/V[-1,-1]
            ###getting world cordinates as last column of V 
            X=V[:4,-1]
            if X[2]>0:
                world_points.append(X[:3])
                
                selected_indices.append(index)
        return world_points, selected_indices


