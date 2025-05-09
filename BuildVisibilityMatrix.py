import numpy as np

def BuildVisibilityMatrix(world_cord_dict, cameraposeList,indexlistcamera):
    # Get matrix dimensions
    I = len(cameraposeList)  # Number of cameras (half of the length of cameraposeList)
    J = len(world_cord_dict)  # Number of world points
    imagelist=[]
    # Initialize visibility matrix with zeros
    Vij = np.zeros((I, J), dtype=int)
    
    for j in range(J):  # Iterate over world points
        pose_set = world_cord_dict[j]  # Get set of cameras seeing this point
        
        for i in range(1, I + 1):  # Iterate over camera poses (1 to I)
            if i in pose_set:
                Vij[i - 1, j] = 1
                imagelist.append(world_cord_dict[j][i])   
                               

    return Vij,imagelist
