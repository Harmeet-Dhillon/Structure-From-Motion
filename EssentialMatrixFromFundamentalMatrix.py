import numpy as np
import math
def EssentialMatrixFromFundamentalMatrix(K1, F, K2):
    E = K1.T @ F @ K2
    print("rank before correction",np.linalg.matrix_rank(E))
    print("E is before ",E)
    U,S, V = np.linalg.svd(E)  
    print("U",U)
    print("S",S)
    print("V",V)
    D = np.diag([1, 1, 0])
    

    E = U @ D @ V # Corrected Essential Matrix
    print("rank after correction",np.linalg.matrix_rank(E))
    print("E is after ",E)
    
    return E
    