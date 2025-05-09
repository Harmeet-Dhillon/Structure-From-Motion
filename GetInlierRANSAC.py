
import copy
import random

import numpy as np
from EstimateFundamentalMatrix import EstimateFundamentalMatrix
from tqdm import tqdm

random.seed(42)

def GetInlierRANSAC(processed_data, M=100, show=False):
    """
    Uses RANSAC to find the best fundamental matrix and prune outliers
    for each image pair.

    Args:
        processed_data (dict): Dictionary containing match data.
        M (int): Number of RANSAC iterations.

    Returns:
        pruned_data (dict): A new dictionary with updated matches.
    """

    pruned_data = copy.deepcopy(processed_data)  # Create a copy to avoid modifying the original data
    epsilon = 0.1  # Epipolar constraint threshold

    for first_image in pruned_data:
        for second_image in pruned_data[first_image]['point_matches_by_image_index']:
            matches = pruned_data[first_image]['point_matches_by_image_index'][second_image]

            len1 = len(matches)  # Store original number of matches
            best_inlier_indices = []
            max_inliers = 0

            # RANSAC loop
            for _ in tqdm(range(1, M + 1), desc=f"[GetInlierRANSAC] Processing image pair ({first_image}, {second_image})"):
                
                # Select N random correspondences
                random_indices = random.sample(range(len(matches)), 8)
                random_subset_matches = [matches[idx] for idx in random_indices]

                # Estimate the Fundamental matrix
                F = EstimateFundamentalMatrix(random_subset_matches)

                # Compute inliers
                current_inlier_indices = []
                for idx, match in enumerate(matches):
                    x1 = np.hstack((np.array(match["current_image_coordinates"]), 1))
                    x2 = np.hstack((np.array(match["matching_image_coordinates"]), 1))

                    if np.abs((x1.T) @ F @ x2) < epsilon:
                        current_inlier_indices.append(idx)

                # Update the best inlier set if it has more inliers
                if len(current_inlier_indices) > max_inliers:
                    max_inliers = len(current_inlier_indices)
                    best_inlier_indices = current_inlier_indices

            # Prune matches based on best inliers
            pruned_data[first_image]['point_matches_by_image_index'][second_image] = [
                matches[idx] for idx in best_inlier_indices
            ]

            len2 = len(pruned_data[first_image]['point_matches_by_image_index'][second_image])
            
            if show:
                print(f"Image ({first_image}, {second_image}): {len1} -> {len2} inliers")

    return pruned_data  # Return modified dictionary
