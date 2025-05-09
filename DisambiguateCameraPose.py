import numpy as np

from LinearTriangulation import LinearTriangulation  # Ensure correct import


def DisambiguateCameraPose(rc_list, K, matching_list):
    bestZscore = 0
    best_pose = None  # Store the best camera pose
    
    for c2, R2 in rc_list:
        c1 = np.array([[0], [0], [0]])  # First camera at the origin
        R1 = np.eye(3)  # Identity rotation

        # Perform triangulation
        Xset,_ = LinearTriangulation(c1, R1, c2, R2, K, matching_list)

        temp_score = 0
        #r3 = R2.T[2, :]  # Third row of R2.T (1×3 vector)
       
        for X in Xset:
            X=np.array(X)
            c2=np.array(c2)
            m=R2.T @ (X.reshape(3,1) - c2.reshape(3,1))
            
            if m[2] > 0:
            # Ensure positive depth in camera coordinates
                temp_score += 1

        # Update best camera pose if this one is better
        if temp_score > bestZscore:
            bestZscore = temp_score
            best_pose = (c2, R2)
    
    return best_pose
###check  if we can give  best points from here only or reapply LinearTriangulation(I,0,bestpose,K,matching_list)
