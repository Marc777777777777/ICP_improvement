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
    Iterative closest point algorithm with a point to point strategy using only a subset of the point cloud.
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



def PCA(points):
    """
    This function will return the eigenvalues and eigenvectors of the covariance matrix of the point cloud.
    """

    barycenter = np.mean(points, axis=0)
    
    cov_mat = (1/points.shape[0])*(points-barycenter).T@(points-barycenter)
    output_eigh = np.linalg.eigh(cov_mat)

    eigenvalues = output_eigh[0]
    eigenvectors = output_eigh[1]

    return eigenvalues, eigenvectors



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


def compute_local_PCA_KNN(query_points, cloud_points, k_queried):
    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points

    # This version is slightly modified to use KNN instead of the radius.

    tree = KDTree(cloud_points, 40)
    neighbors_list = tree.query(query_points, k=k_queried)[1]

    all_eigenvalues = np.zeros((cloud_points.shape[0], 3))
    all_eigenvectors = np.zeros((cloud_points.shape[0], 3, 3))

    for i in range(neighbors_list.shape[0]):
        # We first get the list of neighbor indices
        neighbor_index = neighbors_list[i].flatten()
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
    eps = 1e-6
    all_eigenvalues, all_eigenvectors = compute_local_PCA(query_points, cloud_points, radius)


    normals = all_eigenvectors[:, :, 0]

    verticality = 2*np.arcsin(np.abs(normals@np.array([0,0,1])))/np.pi
    linearity = 1 - all_eigenvalues[:,1]/(all_eigenvalues[:,2]+eps)
    planarity = (all_eigenvalues[:,1] - all_eigenvalues[:,0])/(all_eigenvalues[:,2]+eps)
    sphericity = all_eigenvalues[:,0]/(all_eigenvalues[:,2]+eps)

    return verticality, linearity, planarity, sphericity