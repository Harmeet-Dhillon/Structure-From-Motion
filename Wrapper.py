

#### After disambiguity we got best camera pose ...
##### Now after getting best poses,, we get final X of corresponding points
#### We call them best Xo ....or we can also get it fromm linear triangulation code
######we have R now ... we get projections P1 by R|C ,,,,
####we apply fnn reprojection error ....,,,we minimise it to get best X 
####PNP

####now add third image which has same points , choose 6 points
#### use SVD ,this gives UVt....we can also get Transation of it from slides...\\\\
#### we imply RANSAC--we make diffeent combs of 6 points and choose with maxmim
##### # inliers ...this is C,R
####we again put projection error which gives us better R,C by non linear PnP

#### Bundle adjustments
###will try to finish tomorow


import ast  # For excel to numpy
import os
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from BuildVisibilityMatrix import BuildVisibilityMatrix
from BundleAdjustment import BundleAdjustment
from DisambiguateCameraPose import DisambiguateCameraPose
from EssentialMatrixFromFundamentalMatrix import \
    EssentialMatrixFromFundamentalMatrix
from EstimateFundamentalMatrix import EstimateFundamentalMatrix
from ExtractCameraPose import ExtractCameraPose
from GetInlierRANSAC import GetInlierRANSAC
from helpers.data_processing import get_data
from LinearTriangulation import LinearTriangulation
from NonlinearPnP import NonlinearPnP
from NonlinearTriangulation import NonlinearTriangulation
from PnPRANSAC import ransac_pose_estimation


# utility
def matches_to_tuple_list(matches):

    matching_list = []

    for match in matches:
        point1 = match["current_image_coordinates"]
        point2 = match["matching_image_coordinates"]
        matching_list.append((point1, point2))

    return matching_list




def projectWorldPointsOnImage(K, existing_list, R, C, X, data_dir, img_idx, use_plt=True):
    image_path = os.path.join(data_dir, f"{img_idx}.png")
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Convert world points to homogeneous coordinates
    X = np.hstack((X, np.ones((X.shape[0], 1))))  
    R_ = R.T
    C_ = -R_ @ C
    H = np.hstack((R_, C_.reshape(-1, 1)))  
    P = K @ H

    # Project new points onto the image plane
    img_points = np.array([P @ x.reshape(4, 1) for x in X])  
    img_points = img_points.squeeze(axis=-1)  
    img_points = img_points[:, :2] / img_points[:, 2][:, np.newaxis]  # Normalize

    # Convert existing_list (tuple) to a NumPy array for plotting
    existing_points = np.array(existing_list)

    if use_plt:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Left Image: Existing Points (Green)
        axes[0].imshow(img)
        axes[0].scatter(existing_points[:, 0], existing_points[:, 1], c='g', marker='o', s=10, label="Existing Points")
        axes[0].set_title("Existing Points")
        axes[0].axis("off")

        # Right Image: Projected Points (Red)
        axes[1].imshow(img)
        axes[1].scatter(img_points[:, 0], img_points[:, 1], c='r', marker='o', s=10, label="Projected Points")
        axes[1].set_title("Projected Points")
        axes[1].axis("off")

        plt.show()

    else:
        img1 = img.copy()
        img2 = img.copy()

        # Draw existing points (green) on img1
        for (x, y) in existing_points:
            cv2.circle(img1, (int(x), int(y)), radius=3, color=(0, 255, 0), thickness=-1)

        # Draw projected points (red) on img2
        for (x, y) in img_points:
            cv2.circle(img2, (int(x), int(y)), radius=3, color=(255, 0, 0), thickness=-1)

        # Concatenate images side by side
        combined_img = np.hstack((img1, img2))

        cv2.imshow("Existing Points (Left) | Projected Points (Right)", combined_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return None



def save_to_excel(common_world_points, filtered_current_inlier_matches, filename='output.xlsx'):
    # Prepare lists to store data for DataFrame
    world_points = []
    img1_points = []
    img2_points = []

    # Iterate through common world points and filtered inlier matches
    for world_point, (img1, img2) in zip(common_world_points, filtered_current_inlier_matches):
        world_points.append(world_point)
        img1_points.append(img1)
        img2_points.append(img2)

    # Create a DataFrame from the lists
    data = {
        'World Point': world_points,
        'Image 1 (x, y)': img1_points,
        'Image 2 (x, y)': img2_points
    }

    df = pd.DataFrame(data)

    # Save the DataFrame to an Excel file
    df.to_excel(filename, index=False, engine='openpyxl')



def get_unique_points_with_no_duplicates(filename='output.xlsx'):
    # Step 1: Load the data from Excel, skipping the first row
    df = pd.read_excel(filename, engine='openpyxl', skiprows=0)

    # Step 2: Extract the columns by index (2nd and 3rd columns)
    common_world_points = df.iloc[:, 0].tolist()  # First column: World Point
    img1_points = df.iloc[:, 1].tolist()  # Second column: Image 1 (x, y)
    img2_points = df.iloc[:, 2].tolist()  # Third column: Image 2 (x, y)

    # Step 3: Count occurrences of each column independently
    world_point_counts = Counter(common_world_points)
    img1_point_counts = Counter(img1_points)
    img2_point_counts = Counter(img2_points)

    # Step 4: Track points where no duplicates are present across any column
    unique_world_points = []  # This will store the world points as numpy arrays
    unique_img_matches = []  # This will store tuples of (img1_point, img2_point) as numpy arrays

    # Step 5: Iterate over the lists and add only those without duplicates in any column
    for i in range(len(common_world_points)):
        world_point = common_world_points[i]
        img1_point = img1_points[i]
        img2_point = img2_points[i]

        # Check if the points are unique in their respective columns
        if (world_point_counts[world_point] == 1 and
            img1_point_counts[img1_point] == 1 and
            img2_point_counts[img2_point] == 1):
            
            # Convert the points to numpy arrays and append
            world_point_array = np.fromstring(world_point.strip("[]"), sep=' ')
            img1_point_array = np.array(ast.literal_eval(img1_point))
            img2_point_array = np.array(ast.literal_eval(img2_point))

            unique_world_points.append(world_point_array)
            unique_img_matches.append((img1_point_array, img2_point_array))  # Store as tuple of numpy arrays

    # Step 6: Return the unique world points and the list of image point matches
    return unique_world_points, unique_img_matches






def draw_matches_pnp(img_idx, current_image_points):
    # Load the image
    img = cv2.imread(img_idx)
    
    if img is None:
        print("Error: Unable to load image.")
        return
    
    # Ensure points are in integer format
    for point in current_image_points:
        x, y = int(point[0]), int(point[1])
        cv2.circle(img, (x, y), radius=3, color=(0, 255, 0), thickness=-1)  # Green filled circle

    # Show the image
    cv2.imshow("Image with Points", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def draw_matches(image1, image2, matches, inliers, window_name="Matches"):
    """
    Draws matching points on two images. Inliers are green, outliers are red.
    
    Args:
        image1 (np.array): First image.
        image2 (np.array): Second image.
        matches (list): List of all matches (dict format).
        inliers (set): Indices of inlier matches.
        window_name (str): Name of the output window.
    """
    img1 = image1.copy()
    img2 = image2.copy()

    for idx, match in enumerate(matches):
        x1, y1 = match["current_image_coordinates"]
        x2, y2 = match["matching_image_coordinates"]

        color = (0, 255, 0) if idx in inliers else (0, 0, 255)  # Green for inliers, red for outliers
        radius = 5

        cv2.circle(img1, (int(x1), int(y1)), radius, color, -1)
        cv2.circle(img2, (int(x2), int(y2)), radius, color, -1)

    combined_image = np.hstack((img1, img2))
    cv2.imshow(window_name, combined_image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    print("Window Closed, Continuing Execution...")  # Debug statement
    cv2.destroyAllWindows()  # Close the window after pressing a key

def getCommonWorldPoints(inlier_matches_12, inlier_matches_1c, world_points_12):
    '''
    # inlier_matches_12: inlier matches between image 1 and 2 (tuple of tuples)
    # inlier_matches_1c: inlier matches between image 1 and current image 'c' (tuple of tuples)
    # world_points_12: world points for inlier matches between image 1 and 2
    '''

    pt_idx_matches_short = []
    pt_idx_matches_long = []

    # identify shorter/longer matches
    if len(inlier_matches_1c) < len(inlier_matches_12):
        shorter_matches = inlier_matches_1c
        longer_matches = inlier_matches_12
    else:
        shorter_matches = inlier_matches_12
        longer_matches = inlier_matches_1c

    # iterate over matches with shorter length

    for short_idx, match in enumerate(shorter_matches):
        img_1_pt = match[0]
    
        # Find the index in longer_matches where the first value matches img_1_pt
        for long_idx, long_match in enumerate(longer_matches):
            if long_match[0] == img_1_pt:
                pt_idx_matches_short.append(short_idx)
                pt_idx_matches_long.append(long_idx)
                break  # Stop once the first match is found

    if len(inlier_matches_1c) < len(inlier_matches_12):
        img_1_pt_indices_1c = pt_idx_matches_short
        img_1_pt_indices_12 = pt_idx_matches_long
    else:
        img_1_pt_indices_12 = pt_idx_matches_short
        img_1_pt_indices_1c = pt_idx_matches_long

    # Filter world points and inlier matches 1c
    common_world_points = [world_points_12[i] for i in img_1_pt_indices_12]
    filtered_inlier_matches_1c = [inlier_matches_1c[i] for i in img_1_pt_indices_1c]

    return common_world_points, filtered_inlier_matches_1c

def main():

    total_images = 5

    data_dir = os.path.join(os.getcwd(), "Data")

    # Get all matches for each image
    point_matches_dict, K = get_data(data_directory=data_dir)

    # Prune outliers using RANSAC
    all_inlier_matches = GetInlierRANSAC(point_matches_dict, M=100)

    # Get matches for image pair (1,2)
    matches = point_matches_dict[1]['point_matches_by_image_index'][2]
    inlier_matches_12 = all_inlier_matches[1]['point_matches_by_image_index'][2]
     
    # Get indices of inliers
    inlier_indices = []
    for match in inlier_matches_12:
        inlier_indices.append(matches.index(match))  # Find index in original match list

    # Load images
    img1 = cv2.imread(os.path.join(data_dir, "1.png"))
    img2 = cv2.imread(os.path.join(data_dir, "2.png"))

    # draw_matches(img1, img2, matches, inlier_indices)

    if img1 is None or img2 is None:
        print("Error loading images.")
        return
    
    # Compute Fundamental Matrix using inlier matches
    F_12 = EstimateFundamentalMatrix(inlier_matches_12)
    # print(np.linalg.matrix_rank(F_12))
    E_12 = EssentialMatrixFromFundamentalMatrix(K, F_12, K)
    
    C_R_list = ExtractCameraPose(E_12)

    matching_list = matches_to_tuple_list(inlier_matches_12)
    (C_12_best, R_12_best) = DisambiguateCameraPose(C_R_list, K, matching_list)
    # # print("C and R best ",C_12_best,R_12_best)
    # for crlist in C_R_list:
    #     C_12_best, R_12_best=crlist
    C0=np.zeros((3,1))
    R0=np.eye(3)
    X0, indexlist= LinearTriangulation(np.zeros((3,1)), np.eye(3), C_12_best, R_12_best, K, matching_list)
    
    filtered_inlier_matches_12=[matching_list[i] for i in indexlist] 
    ###################
    X0 = np.array(X0)
    draw_points_1=[match1  for match1,match2 in filtered_inlier_matches_12]
    draw_points_2=[match2  for match1,match2 in filtered_inlier_matches_12]
    # projectWorldPointsOnImage(K,draw_points_1,R0,C0, X0, data_dir, 1, use_plt=False)
    # projectWorldPointsOnImage(K,draw_points_2,R_12_best,C_12_best, X0, data_dir, 2, use_plt=False)
    
    print("X0",X0[0])
    #creating hash map for easy check 
    total_indices_camerawise=[[]]
    total_indices_camerawise.append(indexlist)
    total_indices_camerawise.append(indexlist)
    ##########################################new addition###############helpful in visibility matrix
    image_points=[match2 for match1,match2 in filtered_inlier_matches_12]
    
    
    world_cord_dict = {}

    for i, (match1, match2) in enumerate(filtered_inlier_matches_12):  
        world_cord_dict[i] = {1: match1, 2: match2}
            
    filtered_inlier_record=filtered_inlier_matches_12
    
      
    added_matches=set()
    for match in filtered_inlier_matches_12:
        p1,p2=match
        added_matches.add(p1)
        added_matches.add(p2)
    #################################################
            

    #X = NonlinearTriangulation(K, C_12_best, R_12_best,X0, filtered_inlier_matches_12)
    # print("X",X[0])
    # projectWorldPointsOnImage(K,draw_points_1,R0,C0, X, data_dir, 1, use_plt=False)
    # projectWorldPointsOnImage(K,draw_points_2,R_12_best,C_12_best, X, data_dir, 2, use_plt=False)
    
    #X = np.loadtxt(os.path.join(os.getcwd(), 'non_lin_triang.txt'), delimiter=",")  # Debugging
    # save_comparison_to_excel(X0, X)

    # World points current set to output of LinearTriangulation
    #X_list = X.tolist() #############################################uncomment for non linear
    
    world_points_record=X0
    visited_cameraposes=[1,2]
    cameraposeList=[(np.zeros((3,1)),np.eye(3)),(C_12_best.reshape(3,1),R_12_best)]
    # Iterate from 3rd image to last image - register every image to 1st image
    for img_idx in range (3, total_images+1):

        first_img_idx = 1
        current_img_idx = img_idx
        ##################################################newly added code for multiple iterations
        common_world_points=[]
        filtered_current_inlier_matches=[]
        for prev in visited_cameraposes:
            current_inlier_matches = all_inlier_matches[prev]['point_matches_by_image_index'][current_img_idx]
            cwp, fcim = getCommonWorldPoints(
                                                                filtered_inlier_record,
                                                                matches_to_tuple_list(current_inlier_matches),
                                                                world_points_record)
            common_world_points.extend(cwp)
            filtered_current_inlier_matches.extend(fcim)


        save_to_excel(common_world_points, filtered_current_inlier_matches, filename='matches.xlsx')
        unique_world_points,unique_matches=get_unique_points_with_no_duplicates(filename='matches.xlsx')
        save_to_excel(unique_world_points, unique_matches, filename='matches_final.xlsx')
        current_image_points =  [match[1] for match in unique_matches]
        img_path = os.path.join(data_dir, str(img_idx) + ".png")
        #draw_matches_pnp(img_path,current_image_points)                                                  
        # Register current image to first image using PnP
        
        C_new, R_new,Xinliers,xinliers,new_indices_list = ransac_pose_estimation(unique_world_points,current_image_points, K)
        ######Adding camerapose to the hash map of world-camera relationship used for visibility matrix
        for i,index in enumerate(new_indices_list):
            world_cord_dict[index][img_idx]=tuple(xinliers[i])
        

        C_best, R_best = NonlinearPnP(Xinliers, xinliers, K, C_new, R_new)
        cameraposeList.append((C_best.reshape(3,1),R_best))
      

        ### get the  X between 1 and 3 which is not there in X0 already generate by 1 and 2 ########
        ###get the combination  which is already not constructed on 3D , for these matches , will construct 3D cords
        new_inlier_matches=[]
        for prev in visited_cameraposes:
            inlier_matches_idx = all_inlier_matches[prev]['point_matches_by_image_index'][img_idx] 
            inlier_matches=matches_to_tuple_list(inlier_matches_idx)
            for match in inlier_matches:
                p1,p2=match
                if p1 not in added_matches:
                    new_inlier_matches.append(match)
                    
        ###applying linear triangular triangulation
        Xidx,indices_idx = LinearTriangulation(np.zeros((3,1)), np.eye(3), C_best, R_best, K,new_inlier_matches)
        ####getting matches between 1 and img_idx which will be constructed in 3D
        filtered_inlier_matches_1idx=[new_inlier_matches[i] for i in indices_idx]
        #updating image points for bundle adjustment
        for _,match2 in filtered_inlier_matches_1idx:
            image_points.append(match2)

        new_indices_list.extend(indices_idx)
        total_indices_camerawise.append(new_indices_list)
        ###adding in image_1 cords in visited set 
        for match in filtered_inlier_matches_1idx:
            p1,p2=match
            added_matches.add(p1)
            added_matches.add(p2)
        ###adding cameraposes to set of X for visibility matrix
        for i,(match1,match2) in enumerate(filtered_inlier_matches_1idx):
            world_cord_dict[len(world_points_record)+i]={1:match1,img_idx:match2}
           
        ######extending worldpointrecord and filtered_inlier_record
        
        filtered_inlier_record.extend(filtered_inlier_matches_1idx)

        ####drawing new points actual positions in image
        draw_points=[match[1] for match in filtered_inlier_matches_1idx]
        #draw_matches_pnp(img_path,draw_points)
        #####converting the Xidx into array for input to draw projections
        Xlin=np.array(Xidx, dtype=float)
        projectWorldPointsOnImage(K,draw_points, R_new, C_new, Xlin, data_dir, img_idx, use_plt=False)
        projectWorldPointsOnImage(K,draw_points, R_best, C_best, Xlin, data_dir, img_idx, use_plt=False)  

        ###applying non linear tringulation to non linear points
        X_new_best = NonlinearTriangulation(K, C_best, R_best,Xlin, filtered_inlier_matches_1idx)

       
        world_points_record = np.concatenate([world_points_record, X_new_best], axis=0)

        
        projectWorldPointsOnImage(K,draw_points, R_best, C_best, X_new_best, data_dir, img_idx, use_plt=False)


        ########bundle in the loop
        # Vij,image_cords_observations=BuildVisibilityMatrix(world_cord_dict,cameraposeList,total_indices_camerawise)
        # cameraposes_final,world_points_final=BundleAdjustment(K, Vij, world_points_record, cameraposeList, image_cords_observations)
        # C_final=cameraposes_final[img_idx-1][0]
        # R_final=cameraposes_final[img_idx-1][1]
        # world_points_final=np.array(world_points_final)
        # projectWorldPointsOnImage(K,draw_points, R_final, C_final,world_points_final, data_dir, img_idx, use_plt=False)
        # print("results")
        # cameraposeList=cameraposes_final
        # world_points_record=world_points_final

        visited_cameraposes.append(img_idx)
        ########################################################################
    Vij,image_cords_observations=BuildVisibilityMatrix(world_cord_dict,cameraposeList,total_indices_camerawise)
    cameraposes_final,world_points_final=BundleAdjustment(K, Vij, world_points_record, cameraposeList, image_cords_observations)
    C_final=cameraposes_final[img_idx-1][0]
    R_final=cameraposes_final[img_idx-1][1]
    world_points_final=np.array(world_points_final)
    projectWorldPointsOnImage(K,draw_points, R_final, C_final, world_points_final, data_dir, img_idx, use_plt=False)

    print("results")
    
if __name__ == "__main__":
    main()

    main()

