 # Import numpy package and name it "np"
import numpy as np


def NoSelection(data, dimension_array, entropy_array):
    '''
    Selects all data points
    Inputs: 
        data = (N_data x d) matrix where "N_data" is the number of points and "d" the dimension
    Returns:
        indices = indices of elements array data that are kept
    '''
    indices =  np.arange(data.shape[1])
    return indices

def RandomSelection(data, dimension_array,entropy_array, ratio): 
    '''
    Selects a random subset of the data points
    Inputs: 
        data = (N_data x d) matrix where "N_data" is the number of points and "d" the dimension
        ratio = ratio of points that we need to keep
    Returns:
        indices = indices of elements array data that are kept
    '''
    N_data = data.shape[1]
    nb_point_sampled = int(N_data*ratio)
    indices = np.random.choice(N_data,nb_point_sampled, replace = False)
    return indices

def EntropySelection(data, dimension_array,entropy_array, threshold):
    '''
    Selects points with an entropy superior to threshold
    Inputs: 
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        entropy_array = (N_data) array of the entropy of each point 
        threshold = threshold for the entropy selection
    Returns:
        indices = indices of elements array data that are kept
    '''
    indices = np.where(entropy_array > threshold)[0]
    return indices

def DimensionSelection(data, dimension_array, entropy_array):
    '''
    Selects points with a dimension feature equal to 2
    Inputs: 
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        dimension_array = (N_data) array of the entropy of each point 
    Returns:
        indices = indices of elements array data that are kept
    '''
    indices = np.where(dimension_array == 2)[0]

    return indices

