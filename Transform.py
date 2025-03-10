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


if __name__ == "__main__":
    # Path of the file
    file_path = 'data/bunny.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate x, y, and z in a (N*3) point matrix
    points = np.vstack((data['x'], data['y'], data['z'])).T

    means = np.mean(points, axis=0)
    transformed_points = points - means
    transformed_points /= 2
    transformed_points += means
    transformed_points[:, 1] -= 0.1

    write_ply('data/little_bunny.ply', [transformed_points], ['x', 'y', 'z', 'red', 'green', 'blue'])