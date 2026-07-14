import yaml
import json
import numpy as np

path = '/netscratch/jeetmal/videos/Ameer_full_setup/extrinsic/'
# Define a custom constructor for the 'opencv-matrix' tag
def opencv_matrix_constructor(loader, node):
    value = node.value
    # Create a numpy matrix from the data
    matrix = np.array(value, dtype=np.float64)
    return matrix.reshape((node.style['rows'], node.style['cols']))

# Register the custom constructor for 'opencv-matrix' tag
# yaml.add_constructor('tag:yaml.org,2002:opencv-matrix', opencv_matrix_constructor)

# Load the intrinsics YAML data from 'intri.yml'
intri_path = path + 'intri.yml'
with open(intri_path, 'r') as file:
    intri_data = yaml.safe_load(file)

# Load the extrinsics YAML data from 'extri.yml'
extri_path = path + 'extri.yml'
with open(extri_path, 'r') as file:
    extri_data = yaml.safe_load(file)

# Initialize the result structure
result = {
    "intrinsics": {},
    "world_2_cam": {},
    "dist": {}
}

# Process intrinsics data
for i, name in enumerate(intri_data['names'], 1):
    K_matrix = np.array(intri_data['K_' + str(i)]['data']).reshape(3, 3)
    result["intrinsics"][str(i)] = K_matrix.tolist()

    # Process distortion coefficients from 'intri.yml'
    dist_key = 'dist_' + str(i)
    if dist_key in intri_data:
        dist_coeffs = intri_data[dist_key]['data']
        result["dist"][str(i)] = dist_coeffs

# Process extrinsics data (rotation, translation)
for i in range(1, 7):  # Looping over cameras 1 to 6
    # Extract Rotation and Translation matrices
    Rot_key = f"Rot_{i}"
    T_key = f"T_{i}"
    R_matrix = np.array(extri_data[Rot_key]['data']).reshape(3, 3)
    T_matrix = np.array(extri_data[T_key]['data']).reshape(3, 1)
    
    # Construct the 4x4 transformation matrix for world_to_cam
    world_to_cam_matrix = np.hstack((R_matrix, T_matrix))  # [Rotation | Translation]
    world_to_cam_matrix = np.vstack((world_to_cam_matrix, [0, 0, 0, 1]))  # Add [0, 0, 0, 1]

    # Store the transformation matrix
    result["world_2_cam"][str(i)] = world_to_cam_matrix.tolist()

# Output the final result as a JSON
with open('camera_params.json', 'w') as outfile:
    json.dump(result, outfile, indent=4)

# Optionally, print the result
print(json.dumps(result, indent=4))
