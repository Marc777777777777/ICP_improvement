 # Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree
from ply import write_ply, read_ply

def compute_local_PCA(query_points, cloud_points, radius):
    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points

    tree = KDTree(cloud_points, 40)
    neighbors_list = tree.query_radius(query_points, radius)
    # neighbors_list = tree.query(query_points, k=30)   # KNN VERSION

    all_eigenvalues = np.zeros((query_points.shape[0], 3))
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3))

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
    """
    This function will compute all important geometrical features for each points of query_points.
    """
    all_eigenvalues, all_eigenvectors = compute_local_PCA(query_points, cloud_points, radius)

    # normals = all_eigenvectors[:, :, 0]
    
    a1D, a2D, a3D = compute_aiD(all_eigenvalues)

    aD = np.column_stack((a1D,a2D,a3D))
    standard_deviation = np.sqrt(np.maximum(all_eigenvalues, 0))
    
    d_star = np.argmax(aD,axis = 1) + 1

    V = standard_deviation[:,0] * standard_deviation[:,1] * standard_deviation[:,2]

    Ef = Shannon_Entropy(a1D, a2D, a3D) 

    return all_eigenvalues, all_eigenvectors, aD, d_star, radius, Ef, V
    

def compute_aiD(eigenvalues, eps = 1e-6):
    standard_deviation = np.sqrt(np.maximum(eigenvalues, 0))
    a1D = standard_deviation[2] - standard_deviation[1]
    a2D = standard_deviation[1] - standard_deviation[0]
    a3D = standard_deviation[0]

    # a1D = 1 - standard_deviation[:,1]/(standard_deviation[:,2]+eps)
    # a2D = (standard_deviation[:,1] - standard_deviation[:,0])/(standard_deviation[:,2]+eps)
    # a3D = standard_deviation[:,0]/(standard_deviation[:,2]+eps)
    
    # normalization = a1D + a2D + a3D
    normalization = standard_deviation[2]
    
    a1D *= 1/normalization
    a2D *= 1/normalization
    a3D *= 1/normalization

    return a1D, a2D, a3D

def Shannon_Entropy(a1D, a2D, a3D):
    """
    This function computes and returns the Shannon Entropy for a1D, a2D, a3D
    """
    return -a1D * np.log(a1D + 1e-10) -a2D * np.log(a2D + 1e-10) -a3D * np.log(a3D + 1e-10) 

def compute_optimal_radius(query_points, point_cloud, radius_list):
    """
    For each point of query_points we compute its optimal radius in radius_list.
    We return optimal_radius_list, an array containing for query_point[i] the optimal radius.
    """
    tree = KDTree(point_cloud)
    n = query_points.shape[0]
    optimal_radius_list = np.zeros(n)
    min_entropy = 10000.0

    for i in range(n):
        for r in radius_list:
            neighbors_idx = tree.query_radius(query_points[i].reshape(1,3), r)[0]
            n_neighbor = len(neighbors_idx)
            if n_neighbor < 10:
                continue  # Skip small neighborhoods

            # Compute the PCA and get eigenvalues
            neighbors = point_cloud[neighbors_idx]
            barycenter_neighbor = np.mean(neighbors, axis=0)
            cov_mat = (1/n_neighbor)*(neighbors-barycenter_neighbor).T@(neighbors-barycenter_neighbor)

            output_eigh = np.linalg.eigh(cov_mat)
            eigenvalue = output_eigh[0]
            a1D, a2D, a3D = compute_aiD(eigenvalue)
            entropy = Shannon_Entropy(a1D, a2D, a3D)
            if entropy < min_entropy:
                min_entropy = entropy
                optimal_radius = r

        optimal_radius_list[i] = optimal_radius
        min_entropy = 10000.0

    return optimal_radius_list


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


def best_rigid_transform_weighted(data, ref, weights):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref by taking account of the weights.
    Inputs :
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
         ref = (d x N) matrix where "N" is the number of points and "d" the dimension
         weights   = (N) array where "N" is the number of points
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''

    R = np.eye(data.shape[0])
    T = np.zeros((data.shape[0],1))
    data_barycenter = ((data @ weights) / np.sum(weights)).reshape(3, 1)
    ref_barycenter = ((ref @ weights) / np.sum(weights)).reshape(3, 1)

    Q_data = data -  data_barycenter
    Q_ref = ref - ref_barycenter

    W = np.diag(weights)

    H = Q_data@W@Q_ref.T

    U, _, Vh = np.linalg.svd(H)


    #There is a variation with another matrix in the middle
    R = Vh.T @ U.T

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = Vh.T @ U.T

    
    T = ref_barycenter - R @ data_barycenter
    return R, T

def computeRn(cloud, n):
    tree = KDTree(cloud.T, 40)
    distance_list, _  = tree.query(cloud.T, n)
    return np.mean(distance_list, axis = 1)