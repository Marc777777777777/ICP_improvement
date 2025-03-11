 # Import numpy package and name it "np"
import numpy as np

def NoRejection(data, ref,V_data, V_ref,):
    '''
    Rejects no point pairs
    
    Parameters:
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
    
    Returns:
        Indices = Indices of the remaining points after rejection.
    '''
    indices =  np.arange(data.shape[1])
    return indices 

def EuclidianRejection(data, ref, V_data, V_ref, rejection_ratio):
    '''
    Rejects a percentage of point pairs based on their Euclidean distance from a reference point.
    
    Parameters:
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        rejection_ratio = The percentage of points to reject
    
    Returns:
        Indices = Indices of the remaining points after rejection.
    '''
    N_data = data.shape[1]
    nb_rejected_points = int(N_data * rejection_ratio)

    dist = np.linalg.norm(data - ref, axis = 0) 
    sorted_indices_descending = np.argsort(dist)[::-1]
    indices = sorted_indices_descending[nb_rejected_points:]
    return indices

def OmnivarianceRejection(data, ref, V_data, V_ref, rejection_ratio):
    '''
    Rejects a percentage of point pairs based on their omnivariance-based distance from a reference point.
    
    Parameters:
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        V_data = (N_data) array of omnivariance of each point of the first dataset
        V_ref = (N_data) array of omnivariance of each point of the reference dataset
        rejection_ratio = The percentage of points to reject
    
    Returns:
        Indices = Indices of the remaining points after rejection.
    '''

    N_data = data.shape[1]
    nb_rejected_points = int(N_data * rejection_ratio)

    dist = np.abs(V_data - V_ref)
    sorted_indices_descending = np.argsort(dist)[::-1]
    indices = sorted_indices_descending[nb_rejected_points:]
    return indices

def DeviationRejection(data, ref, V_data, V_ref):
    '''
    Rejects point pairs if they have a distance superior to 2.5 the standart deviation
    
    Parameters:
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        V = (N_data) array of omnivariance of each point 
    
    Returns:
        Indices = Indices of the remaining points after rejection.
    '''

    dist = np.linalg.norm(data - ref, axis = 0) 
    deviation = np.std(dist)
    indices = np.where(dist < 2.5 * deviation)[0]

    return indices


