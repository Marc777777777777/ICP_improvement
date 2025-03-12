 # Import numpy package and name it "np"
import numpy as np


def ConstantWeighting(data, ref, normal_data, normal_ref, V_data, V_ref):
    '''
    Weights point pairs with a constant value
    Inputs: 
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
    Returns:
        w = weights of each point pair
    '''
    N_data = data.shape[1]
    w = np.ones(N_data)
    return w

def OmnivarianceWeighting(data, ref, normal_data, normal_ref, V_data, V_ref):
    '''
    Weights point pairs based on the omnivariance distance
    Inputs: 
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        V_data = (N_data) array of omnivariance of each point of the first dataset
        V_ref = (N_data) array of omnivariance of each point of the reference dataset
    Returns:
        w = weights of each point pair
    '''

    dist = np.abs(V_data - V_ref)
    max_dist = np.max(dist)
    w = 1 - dist/max_dist
    return w

def NormalWeighting(data, ref, normal_data, normal_ref, V_data, V_ref):
    '''
    Weights point pairs based on the omnivariance distance
    Inputs: 
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        normal_data = (d x N_data) matrix of normals of each point of the first dataset
        normal_ref = (d x N_data) matrix of normals of each point of the reference dataset
    Returns:
        w = weights of each point pair
    '''

    w = np.sum(normal_data * normal_ref, axis=1)
    return w

