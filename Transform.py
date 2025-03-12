# Import numpy package and name it "np"
import numpy as np

# Import library to plot in python
from matplotlib import pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply
from visu import show_ICP

import sys

def rotation_matrix_x(theta):
    """
    Generates a rotation matrix for a rotation around the X-axis by angle theta.
    
    :param theta: Rotation angle in radians.
    :return: (3, 3) rotation matrix.
    """
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

def rotation_matrix_y(theta):
    """
    Generates a rotation matrix for a rotation around the Y-axis by angle theta.
    
    :param theta: Rotation angle in radians.
    :return: (3, 3) rotation matrix.
    """
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def rotation_matrix_z(theta):
    """
    Generates a rotation matrix for a rotation around the Z-axis by angle theta.
    
    :param theta: Rotation angle in radians.
    :return: (3, 3) rotation matrix.
    """
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

if __name__ == "__main__":
    # Path of the file
    file_path = 'data/bunny.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate x, y, and z in a (N*3) point matrix
    points = np.vstack((data['x'], data['y'], data['z'])).T
    index_sampled = np.random.choice(points.shape[0], size=int(0.8*points.shape[0]), replace=False)
    points = points[index_sampled]
    # We select a subsample of the original point cloud
    # points = np.random.choice(points, size=int(0.9*points.shape[0]))

    # means = np.mean(points, axis=0)

    # Small translation
    # translation = 0.01
    translation = np.random.normal(scale=0.01, size =(1,3))
    transformed_points = points + translation

    # Small rotation
    theta_x = np.radians(10)  # Rotate by 20 degrees around X-axis
    # theta_y = np.radians(5)  # Rotate by 15 degrees around Y-axis
    theta_z = np.radians(10)

    R_x = rotation_matrix_x(theta_x)
    # R_y = rotation_matrix_y(theta_y)
    R_z = rotation_matrix_z(theta_z)
    R = R_x@R_z  # First rotate around X, then around Y, then around Z
    transformed_points = transformed_points@R

    write_ply('data/modified_bunny.ply', [transformed_points], ['x', 'y', 'z'])

