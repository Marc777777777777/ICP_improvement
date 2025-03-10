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

def icp_point_to_point(data, ref, max_iter, RMS_threshold):
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
    RMS_list = []

    it = 0
    RMS = np.inf
    tree = KDTree(ref.T, leaf_size=35)

    while it < max_iter and RMS > RMS_threshold:
        matching_neighbour = tree.query(data_aligned.T)[1].flatten()
        R, T = best_rigid_transform(data_aligned, ref[:, matching_neighbour])
        data_aligned = R@data_aligned + T

        distance = np.sum(np.power(data_aligned - ref[:, matching_neighbour], 2), axis=0)
        RMS = np.sqrt(np.mean(distance))

        it+=1
        
        if(len(R_list)==0):
            R_list.append(R)
            T_list.append(T)
        else:
            R_list.append(R@R_list[-1])
            T_list.append(R@T_list[-1]+T)
        neighbors_list.append(matching_neighbour)
        RMS_list.append(RMS)

    return data_aligned, R_list, T_list, neighbors_list, RMS_list

def icp_point_to_point_fast(data, ref, max_iter, RMS_threshold, sampling_limit):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
        sampling_limit = number of point used at each iteration
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
    RMS_list = []

    it = 0
    RMS = np.inf
    tree = KDTree(ref.T, leaf_size=35)
    N_data = data.shape[1]

    while it < max_iter and RMS > RMS_threshold:
        if N_data > sampling_limit:
            idx = np.random.choice(N_data,sampling_limit,replace=False)
            data_used = data_aligned[:,idx]
        else:
            data_used = data_aligned
        matching_neighbour = tree.query(data_used.T)[1].flatten()
        R, T = best_rigid_transform(data_used, ref[:, matching_neighbour])
        
        data_aligned = R@data_aligned + T
        if N_data > sampling_limit:
            distance = np.sum(np.power(data_aligned[:,idx] - ref[:, matching_neighbour], 2), axis=0)
            RMS = np.sqrt(np.mean(distance))
        else:
            distance = np.sum(np.power(data_aligned - ref[:, matching_neighbour], 2), axis=0)
            RMS = np.sqrt(np.mean(distance))

        it+=1
        
        if(len(R_list)==0):
            R_list.append(R)
            T_list.append(T)
        else:
            R_list.append(R@R_list[-1])
            T_list.append(R@T_list[-1]+T)
        neighbors_list.append(matching_neighbour)
        RMS_list.append(RMS)

    return data_aligned, R_list, T_list, neighbors_list, RMS_list


def icp_ultimate(data, ref, max_iter, RMS_threshold, Selection = NoSelection, Weighting = ConstantWeighting, Rejection = NoRejection):
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
    RMS_list = []

    it = 0
    RMS = np.inf
    tree = KDTree(ref.T, leaf_size=35)

    #radius = 0.05
    radius_list = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0,10.0]
    radius_list_data = compute_optimal_radius(data.T, data.T, radius_list) 
    radius_list_ref = compute_optimal_radius(ref.T, ref.T, radius_list) 

    all_eigenvalues_data, all_eigenvectors_data, aD_data, d_star_data, radius_data, Ef_data, V_data = compute_features(data.T, data.T, radius_list_data)
    all_eigenvalues_ref, all_eigenvectors_ref, aD_ref, d_star_ref, radius_ref, Ef_ref, V_ref = compute_features(ref.T, ref.T, radius_list_ref)

    while it < max_iter and RMS > RMS_threshold:

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


        valid_indices = selected_points_idx[non_rejected_points_idx]  # retrieving the indexes of the original points
        distance = np.sum(np.power(data_aligned[:,valid_indices] - ref[:, non_rejected_matching_neighbour], 2), axis=0)
        RMS = np.sqrt(np.mean(distance))

        it+=1
        
        if(len(R_list)==0):
            R_list.append(R)
            T_list.append(T)
        else:
            R_list.append(R@R_list[-1])
            T_list.append(R@T_list[-1]+T)

        neighbors_list.append(matching_neighbour)
        RMS_list.append(RMS)

    return data_aligned, R_list, T_list, neighbors_list, RMS_list


#------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#


if __name__ == '__main__':

    if False:         
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z']))
        
#selecting_function =  NoSelection
        selecting_function = lambda data, d_star, Ef: RandomSelection(data,d_star, Ef, 0.1)
        #selecting_function = lambda data, d_star, Ef: EntropySelection(data,d_star, Ef, 0.6)
        #selecting_function = lambda data, d_star, Ef:: EntropySelection(data,d_star,Ef, 0.7)
        #selecting_function = DimensionSelection

        #weighting_function = ConstantWeighting
        #weighting_function = OmnivarianceWeighting
        weighting_function = NormalWeighting

        #rejection_function = NoRejection
        #rejection_function = lambda data, ref, V_data, V_ref : EuclidianRejection(data,ref,V_data, V_ref, 0.7)
        #rejection_function = lambda data, ref, V_data, V_ref : OmnivarianceRejection(data,ref,V_data, V_ref, 0.5)
        #rejection_function = lambda data, ref, V_data, V_ref : OmnivarianceRejection(data,ref,V_data, V_ref, 0.7)
        #rejection_function = lambda data, ref, V_data, V_ref : OmnivarianceRejection(data,ref,V_data, V_ref, 0.9)
        rejection_function = DeviationRejection



        new_cloud, R_list, T_list, neighbors_list, RMS_list = icp_ultimate(cloud, cloud, 100, 0.6, selecting_function, weighting_function, rejection_function)

        #show_ICP(cloud, cloud, R_list, T_list, neighbors_list)
   
    # Transformation estimation
    # *************************
    #

    # If statement to skip this part if wanted
    if True:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_r_path = '../data/bunny_returned.ply'

		# Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_r_ply = read_ply(bunny_r_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_r = np.vstack((bunny_r_ply['x'], bunny_r_ply['y'], bunny_r_ply['z']))
        

        selecting_function =  NoSelection
        #selecting_function = lambda data, d_star, Ef: RandomSelection(data,d_star, Ef, 0.1)
        #selecting_function = lambda data, d_star, Ef: EntropySelection(data,d_star, Ef, 0.6)
        #selecting_function = lambda data, d_star, Ef:: EntropySelection(data,d_star,Ef, 0.7)
        #selecting_function = DimensionSelection

        #weighting_function = ConstantWeighting
        weighting_function = OmnivarianceWeighting
        #weighting_function = NormalWeighting

        #rejection_function = NoRejection
        rejection_function = lambda data, ref, V_data, V_ref : EuclidianRejection(data,ref,V_data, V_ref, 0.7)
        #rejection_function = lambda data, ref, V_data, V_ref : OmnivarianceRejection(data,ref,V_data, V_ref, 0.5)
        #rejection_function = lambda data, ref, V_data, V_ref : OmnivarianceRejection(data,ref,V_data, V_ref, 0.7)
        #rejection_function = lambda data, ref, V_data, V_ref : OmnivarianceRejection(data,ref,V_data, V_ref, 0.9)
        #rejection_function = DeviationRejection


        bunny_r_opt, R_list, T_list, neighbors_list, RMS_list = icp_ultimate(bunny_r, bunny_o, 100, 1e-4, selecting_function, weighting_function, rejection_function)
        
        show_ICP(bunny_r, bunny_o, R_list, T_list, neighbors_list)


        # Find the best transformation
        #R, T = best_rigid_transform(bunny_r, bunny_o)

        # Apply the tranformation
        #bunny_r_opt = R.dot(bunny_r) + T

        # Save cloud
        #write_ply('../bunny_r_opt', [bunny_r_opt.T], ['x', 'y', 'z'])

        # Compute RMS
        distances2_before = np.sum(np.power(bunny_r - bunny_o, 2), axis=0)
        RMS_before = np.sqrt(np.mean(distances2_before))
        distances2_after = np.sum(np.power(bunny_r_opt - bunny_o, 2), axis=0)
        RMS_after = np.sqrt(np.mean(distances2_after))

        print('Average RMS between points :')
        print('Before = {:.3f}'.format(RMS_before))
        print(' After = {:.3f}'.format(RMS_after))
   

    # Test ICP and visualize
    # **********************
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        ref2D_path = '../data/ref2D.ply'
        data2D_path = '../data/data2D.ply'
        
        # Load clouds
        ref2D_ply = read_ply(ref2D_path)
        data2D_ply = read_ply(data2D_path)
        ref2D = np.vstack((ref2D_ply['x'], ref2D_ply['y']))
        data2D = np.vstack((data2D_ply['x'], data2D_ply['y']))        

        compute_features(ref2D, ref2D, 0.1)


        # Apply ICP
        #data2D_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(data2D, ref2D, 10, 1e-4)
        
        # Show ICP
        #show_ICP(data2D, ref2D, R_list, T_list, neighbors_list)
        
        # Plot RMS
        #plt.plot(RMS_list)
        #plt.xlabel("Itération")
        #plt.ylabel("RMS")
        #plt.show()
    

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_p_path = '../data/bunny_perturbed.ply'
        
        # Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_p_ply = read_ply(bunny_p_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_p = np.vstack((bunny_p_ply['x'], bunny_p_ply['y'], bunny_p_ply['z']))

        # Apply ICP
        #bunny_p_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(bunny_p, bunny_o, 25, 1e-4)
        
        # Show ICP
        #show_ICP(bunny_p, bunny_o, R_list, T_list, neighbors_list)
        
        # Plot RMS
        #plt.plot(RMS_list)
        #plt.xlabel("Itération")
        #plt.ylabel("RMS")
        #plt.show()

# If statement to skip this part if wanted
    if False:

        # Cloud paths
        notredame_o_path = '../data/Notre_Dame_Des_Champs_1.ply'
        notredame_p_path = '../data/Notre_Dame_Des_Champs_2.ply'
        
        # Load clouds
        notredame_o_ply = read_ply(notredame_o_path)
        notredame_ply = read_ply(notredame_p_path)
        notredame_o = np.vstack((notredame_o_ply['x'], notredame_o_ply['y'], notredame_o_ply['z']))
        notredame_p = np.vstack((notredame_ply['x'], notredame_ply['y'], notredame_ply['z']))

        # Apply ICP
        notredame_p_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point_fast(notredame_p, notredame_o, 25, 1e-4,10000)
        
        # Show ICP
        show_ICP(notredame_p, notredame_o, R_list, T_list, neighbors_list)
        
        # Plot RMS
        plt.plot(RMS_list)
        plt.xlabel("Itération")
        plt.ylabel("RMS")
        plt.show()