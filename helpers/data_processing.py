import os
import re

import numpy as np


def read_feature_matching_files(data_dir):
    file_pattern = re.compile(r"matching(\d+)\.txt$") # d+ is for one or more digits
    data_store = {}

    for file in os.listdir(data_dir):
        match = file_pattern.match(file)
        if match:
            index = int(match.group(1))
            file_path = os.path.join(data_dir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                data_store[index] = f.readlines()

    return data_store

def process_data(data_store):

    ''''
    
    
    Returns:
        - image_index_1 (int): Gives "mini_dict" for first image

            where mini_dict is formated as:
                - 'nFeatures' (int): number of features for this image in total
                - 'point_matches_by_image_index' (dict): dictionary of match information


                where "point_matches_by_image_index" is formatted as:
                    - 'second_image_index' (int): gives match in the form of a dictionary:
                        - 'num_matches' (int): Number of matches for this location in first image
                        - 'color' (tuple of float): RGB color values (r, g, b).
                        - 'current_image_coordinates' (tuple of float): Coordinates (u_image_1, v_image_1) in the first image.
                        - 'matching_image_coordinates' (tuple of float): Corresponding coordinates (u_image_2, v_image_2) in the second image.
    '''

    processed_data = {}

    for key, value in data_store.items():
        # Dictionary to store point pairs by image_id
        point_matches_by_image_index = {}
        
        # The first string contains the number of features
        nFeatures_line = value[0]
        nFeatures = int(nFeatures_line.split(":")[1].strip())  # Extract number of features
        
        # Process each subsequent line containing matches
        for line in value[1:]:
            # Split the line into components
            parts = line.split()
            
            # Extract feature match information
            num_matches = int(parts[0])  # Number of matches for the jth feature
            red_value = int(parts[1])  # Red value
            green_value = int(parts[2])  # Green value
            blue_value = int(parts[3])  # Blue value
            u_current_image = float(parts[4])  # u-coordinate in the current image
            v_current_image = float(parts[5])  # v-coordinate in the current image
            
            # Parse pattern_2 (additional repetitions)
            pattern_2 = []
            i = 6 # New index starting after v_current_image
            while i < len(parts):
                image_id = int(parts[i])  # Image ID
                u_matching_image = float(parts[i+1])  # u-coordinate for the given image ID
                v_matching_image = float(parts[i+2])  # v-coordinate for the given image ID
                pattern_2.append((image_id, u_matching_image, v_matching_image))
                i += 3  # Move to the next set of pattern_2

                if image_id not in point_matches_by_image_index:
                    point_matches_by_image_index[image_id] = []

                point_matches_by_image_index[image_id].append({
                        'num_matches': num_matches,
                        'color': (red_value, green_value, blue_value),
                        'current_image_coordinates': (u_current_image, v_current_image),
                        'matching_image_coordinates': (u_matching_image, v_matching_image)
                    })
            
            # # Store the processed data
            # processed_key_data.append({
            #     'num_matches': num_matches,
            #     'color': (red_value, green_value, blue_value),
            #     'current_image_coordinates': (u_current_image, v_current_image),
            #     'pattern_2': pattern_2
            # })
        
        # Store the processed data for the current key
        processed_data[key] = {
            'nFeatures': nFeatures,
            # 'matches': processed_key_data,
            'point_matches_by_image_index': point_matches_by_image_index
        }
    
    return processed_data


def get_data(data_directory):
    data_store = read_feature_matching_files(data_directory)
    processed_data = process_data(data_store)

    K = np.loadtxt(os.path.join(data_directory, "calibration.txt"))

    return processed_data, K