import numpy as np

def ExtractCameraPose(E):
    U, S, Vt = np.linalg.svd(E)

    c1 = U[:, 2]  # First possible translation
    c2 = -U[:, 2] # Second possible translation

    # Rotation matrices
    W = np.array([[0, -1, 0], 
                  [1,  0, 0], 
                  [0,  0, 1]])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    if np.linalg.det(R1)==-1:
        R1=-R1    ####R is camera rotation w.r.t to world
        c1=-c1
        c2=-c2
    if np.linalg.det(R2)==-1:
        R2=-R2
        c2=-c2
        c1=-c1
    return [(c1,R1),(c1,R2),(c2,R1),(c2,R2)]

    