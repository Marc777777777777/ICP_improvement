 
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


#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#
def compute_local_PCA(query_points, cloud_points, radius):
    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points

    tree = KDTree(cloud_points, 40)
    neighbors_list = tree.query_radius(query_points, radius)
    # neighbors_list = tree.query(query_points, k=30)   # TODO: FOR QUESTION 4 uncommment the line and comment the one above

    all_eigenvalues = np.zeros((cloud_points.shape[0], 3))
    all_eigenvectors = np.zeros((cloud_points.shape[0], 3, 3))

    for i in range(neighbors_list.shape[0]):
        # We first get the list of neighbor indices
        neighbor_index = neighbors_list[i]
        n_neighbor = neighbor_index.shape[0]
        neighbors = cloud_points[neighbor_index, :]

        # Then we compute the PCA and get the eigenvalues and eigenvectors
        barycenter_neighbor = np.mean(neighbors, axis=0)
        cov_mat = (1/n_neighbor)*(neighbors-barycenter_neighbor).T@(neighbors-barycenter_neighbor)
        output_eigh = np.linalg.eigh(cov_mat)
        eigenvalue = output_eigh[0]
        eigenvector = output_eigh[1]

        all_eigenvalues[i] = eigenvalue
        all_eigenvectors[i] = eigenvector

    return all_eigenvalues, all_eigenvectors
    

def compute_features(query_points, cloud_points, radius):
    n = query_points.shape[0]
    eps = 1e-6
    all_eigenvalues, all_eigenvectors = compute_local_PCA(query_points, cloud_points, radius)

    print("Min eigenvalue:", np.min(all_eigenvalues))

    normals = all_eigenvectors[:, :, 0]
    
    standart_deviation = np.sqrt(np.maximum(all_eigenvalues, 0))


    a1D = 1 - standart_deviation[:,1]/(standart_deviation[:,2]+eps)
    a2D = (standart_deviation[:,1] - standart_deviation[:,0])/(standart_deviation[:,2]+eps)
    a3D = standart_deviation[:,0]/(standart_deviation[:,2]+eps)
    
    sum_ad = a1D + a2D + a3D
    
    a1D *= 1/sum_ad
    a2D *= 1/sum_ad
    a3D *= 1/sum_ad

    aD = np.column_stack((a1D,a2D,a3D))

    
    d_star = np.argmax(aD,axis = 1) + 1

    V = standart_deviation[:,0] * standart_deviation[:,1] * standart_deviation[:,2]

    Ef  = -a1D * np.log(a1D + 1e-10) -a2D * np.log(a2D + 1e-10) -a3D * np.log(a3D + 1e-10) 

    return all_eigenvalues, all_eigenvectors, aD, d_star, radius, Ef, V

def best_rigid_transform(data, ref):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
         ref = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''

    # YOUR CODE
    R = np.eye(data.shape[0])
    T = np.zeros((data.shape[0],1))
    data_barycenter = np.mean(data, axis = 1, keepdims = True)
    ref_barycenter = np.mean(ref, axis = 1, keepdims = True)

    Q_data = data -  data_barycenter
    Q_ref = ref - ref_barycenter

    H = Q_data@Q_ref.T

    U, _, Vh = np.linalg.svd(H)


    R = Vh.T @ U.T

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = Vh.T @ U.T

    
    T = ref_barycenter - R @ data_barycenter
    return R, T



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



#------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#


if __name__ == '__main__':

    if True:         
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
        all_eigenvalues, all_eigenvectors, aD, d_star, radius, Ef, V = compute_features(cloud, cloud, 0.50)
   
    # Transformation estimation
    # *************************
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_r_path = '../data/bunny_returned.ply'

		# Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_r_ply = read_ply(bunny_r_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_r = np.vstack((bunny_r_ply['x'], bunny_r_ply['y'], bunny_r_ply['z']))

        # Find the best transformation
        R, T = best_rigid_transform(bunny_r, bunny_o)

        # Apply the tranformation
        bunny_r_opt = R.dot(bunny_r) + T

        # Save cloud
        write_ply('../bunny_r_opt', [bunny_r_opt.T], ['x', 'y', 'z'])

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
        