# Import numpy package and name it "np"
import numpy as np
import json
import os
import time

# Import library to plot in python
from matplotlib import pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply
from visu import show_ICP

import sys

# Import functions from all the differents ICP steps
from ICP_features import *
from ICP_selection import *
from ICP_rejection import *
from ICP_weighting import *

#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#

def icp_ultimate(data, ref, max_iter, distance_threshold, filename, Selection = NoSelection, Weighting = ConstantWeighting, Rejection = NoRejection):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
           
    '''
    # Variable for aligned data
    data_aligned = np.copy(data)

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    distance_list = []
    time_list = []

    it = 0
    distance = np.inf
    tree = KDTree(ref.T, leaf_size=35)

    if os.path.exists(filename):
        with open(filename, 'r') as f:
            loaded_results = json.load(f)
        print("Features loaded from the JSON file")

        all_eigenvalues_data = np.array(loaded_results['all_eigenvalues_data'])
        all_eigenvectors_data = np.array(loaded_results['all_eigenvectors_data'])
        aD_data = np.array(loaded_results['aD_data'])
        d_star_data = np.array(loaded_results['d_star_data'])
        radius_data = np.array(loaded_results['radius_data'])
        Ef_data = np.array(loaded_results['Ef_data'])
        V_data = np.array(loaded_results['V_data'])

        all_eigenvalues_ref = np.array(loaded_results['all_eigenvalues_ref'])
        all_eigenvectors_ref = np.array(loaded_results['all_eigenvectors_ref'])
        aD_ref = np.array(loaded_results['aD_ref'])
        d_star_ref = np.array(loaded_results['d_star_ref'])
        radius_ref = np.array(loaded_results['radius_ref'])
        Ef_ref = np.array(loaded_results['Ef_ref'])
        V_ref = np.array(loaded_results['V_ref'])

    else:
        radius_list = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        radius_list_data = compute_optimal_radius(data.T, data.T, radius_list) 
        radius_list_ref = compute_optimal_radius(ref.T, ref.T, radius_list) 

        print("Optimal radii calculated")

        all_eigenvalues_data, all_eigenvectors_data, aD_data, d_star_data, radius_data, Ef_data, V_data = compute_features(data.T, data.T, radius_list_data)
        all_eigenvalues_ref, all_eigenvectors_ref, aD_ref, d_star_ref, radius_ref, Ef_ref, V_ref = compute_features(ref.T, ref.T, radius_list_ref)
        
        data_to_save = {
            'all_eigenvalues_data': all_eigenvalues_data.tolist(),
            'all_eigenvectors_data': all_eigenvectors_data.tolist(),
            'aD_data': aD_data.tolist(),
            'd_star_data': d_star_data.tolist(),
            'radius_data': radius_data.tolist(),
            'Ef_data': Ef_data.tolist(),
            'V_data': V_data.tolist(),
            'all_eigenvalues_ref': all_eigenvalues_ref.tolist(),
            'all_eigenvectors_ref': all_eigenvectors_ref.tolist(),
            'aD_ref': aD_ref.tolist(),
            'd_star_ref': d_star_ref.tolist(),
            'radius_ref': radius_ref.tolist(),
            'Ef_ref': Ef_ref.tolist(),
            'V_ref': V_ref.tolist()
        }

        with open(filename, 'w') as f:
            json.dump(data_to_save, f)
        print("Data saved successfully")

    n = 5
    Rn = computeRn(data, n)

    start_time = time.time()

    while it < max_iter and distance > distance_threshold:
        if it%10 == 0:
            print("Itération", it)
        # Selection
        selected_points_idx = Selection(data_aligned, d_star_data, Ef_data)
        data_selected = data_aligned[:,selected_points_idx]

        # Matching
        matching_neighbour = tree.query(data_selected.T)[1].flatten()

        # Weighting
        V_data_selected = V_data[selected_points_idx]     #Keep the V of the new points only
        V_ref_selected = V_ref[matching_neighbour]

        normal_data_selected = all_eigenvectors_data[selected_points_idx, :, 0] #Same
        normal_ref_selected = all_eigenvectors_ref[matching_neighbour, :, 0] 

        # Need to verify order of eigenvector:
        weights_selected = Weighting(data_selected,ref[:,matching_neighbour], normal_data_selected,normal_ref_selected, V_data_selected, V_ref_selected)

        # Rejection
        non_rejected_points_idx = Rejection(data_selected, ref[:,matching_neighbour], V_data_selected, V_ref_selected)
        data_non_rejected = data_selected[:,non_rejected_points_idx]     #Update the points and their matched ones
        non_rejected_matching_neighbour = matching_neighbour[non_rejected_points_idx]

        # Minimizing
        non_rejected_weights = weights_selected[non_rejected_points_idx]

        R, T = best_rigid_transform_weighted(data_non_rejected, ref[:, non_rejected_matching_neighbour], non_rejected_weights)
        data_aligned = R@data_aligned + T

        #calculating distance
        distance_matching = tree.query(data_aligned.T)[0].flatten()
        distances_kept = distance_matching[distance_matching< 10 * Rn]
        if len(distances_kept) != 0:    
            distance =  np.mean(distances_kept)
        else:
            distance = np.inf

        it+=1
        
        if(len(R_list)==0):
            R_list.append(R)
            T_list.append(T)
        else:
            R_list.append(R@R_list[-1])
            T_list.append(R@T_list[-1]+T)

        neighbors_list.append(matching_neighbour)
        distance_list.append(distance)
        time_list.append(time.time()-start_time)

    return data_aligned, R_list, T_list, neighbors_list, time_list, distance_list


