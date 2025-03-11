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
from ICP import *

if __name__ == '__main__':
    # Cloud paths
    #Street
    #cloud_o_path = '../data/modified_street.ply'
    #cloud_r_path = '../data/Lille_street_small_normals.ply'
    #filename = '../stored_features/street.json'             #to store features 

    #Bunny
    cloud_o_path = '../data/bunny_original.ply'
    cloud_r_path = '../data/bunny_returned.ply'
    filename = '../stored_features/bunny.json'

	# Load clouds
    cloud_o_ply = read_ply(cloud_o_path)
    cloud_r_ply = read_ply(cloud_r_path)
    cloud_o = np.vstack((cloud_o_ply['x'], cloud_o_ply['y'], cloud_o_ply['z']))
    cloud_r = np.vstack((cloud_r_ply['x'], cloud_r_ply['y'], cloud_r_ply['z']))
        

    selecting_function =  NoSelection
    #selecting_function = lambda data, d_star, Ef: RandomSelection(data,d_star, Ef, 0.1)
    #selecting_function = lambda data, d_star, Ef: EntropySelection(data,d_star, Ef, 0.6)
    #selecting_function = lambda data, d_star, Ef: EntropySelection(data,d_star,Ef, 0.7)
    #selecting_function = DimensionSelection

    weighting_function = ConstantWeighting
    #weighting_function = OmnivarianceWeighting
    #weighting_function = NormalWeighting

    rejection_function = NoRejection
    #rejection_function = lambda data, ref, V_data, V_ref : EuclidianRejection(data,ref,V_data, V_ref, 0.7)
    #rejection_function = lambda data, ref, V_data, V_ref : OmnivarianceRejection(data,ref,V_data, V_ref, 0.5)
    #rejection_function = lambda data, ref, V_data, V_ref : OmnivarianceRejection(data,ref,V_data, V_ref, 0.7)
    #rejection_function = lambda data, ref, V_data, V_ref : OmnivarianceRejection(data,ref,V_data, V_ref, 0.9)
    #rejection_function = DeviationRejection

    selecting_function_list = [
        NoSelection,
        lambda data, d_star, Ef: RandomSelection(data,d_star, Ef, 0.1),
        lambda data, d_star, Ef: EntropySelection(data,d_star, Ef, 0.6),
        lambda data, d_star, Ef: EntropySelection(data,d_star,Ef, 0.7),
        DimensionSelection
    ]

    weighting_function_list = [
        ConstantWeighting,
        OmnivarianceWeighting,
        NormalWeighting
    ]

    rejection_function_list = [
        NoRejection,
        lambda data, ref, V_data, V_ref : EuclidianRejection(data,ref,V_data, V_ref, 0.7),
        lambda data, ref, V_data, V_ref : OmnivarianceRejection(data,ref,V_data, V_ref, 0.5),
        lambda data, ref, V_data, V_ref : OmnivarianceRejection(data,ref,V_data, V_ref, 0.7),
        lambda data, ref, V_data, V_ref : OmnivarianceRejection(data,ref,V_data, V_ref, 0.9),
        DeviationRejection
    ]

    iteration = 30
    tol = 1e-4

    if False:
        cloud_r_opt, R_list, T_list, neighbors_list, distance_list = icp_ultimate(cloud_r, cloud_o, iteration, tol, filename, selecting_function, weighting_function, rejection_function)

        show_ICP(cloud_r, cloud_o, R_list, T_list, neighbors_list)

        # Compute RMS
        distances2_before = np.sum(np.power(cloud_r - cloud_o, 2), axis=0)
        RMS_before = np.sqrt(np.mean(distances2_before))
        distances2_after = np.sum(np.power(cloud_r_opt - cloud_o, 2), axis=0)
        RMS_after = np.sqrt(np.mean(distances2_after))

        print('Average RMS between points :')
        print('Before = {:.3f}'.format(RMS_before))
        print(' After = {:.3f}'.format(RMS_after))

        plt.plot(range(len(distance_list)), distance_list)
        plt.xlabel("Iterations")
        plt.ylabel("Distance")
        plt.title("Evolution of distances")
        plt.show()    

    if True:
        label_list = ["Default", "Random","Ef > 0.6", "Ef > 0.7", "d*=2"]
        for i,s_function in enumerate(selecting_function_list):
            cloud_r_opt, R_list, T_list, neighbors_list, distance_list = icp_ultimate(cloud_r, cloud_o, iteration, tol, filename, s_function, weighting_function, rejection_function)
            plt.plot(range(len(distance_list)), distance_list,'k-', label = label_list[i])

        plt.xlabel("Iterations")
        plt.ylabel("Distance")
        plt.title("Evolution of distances")
        plt.legend()
        plt.show() 

    if False:
        label_list = ["Default", "Omnivariance","Normal"]
        for i,w_function in enumerate(weighting_function_list):
            cloud_r_opt, R_list, T_list, neighbors_list, distance_list = icp_ultimate(cloud_r, cloud_o, iteration, tol, filename, selecting_function, w_function, rejection_function)
            plt.plot(range(len(distance_list)), distance_list,'k-', label = label_list[i])

        plt.xlabel("Iterations")
        plt.ylabel("Distance")
        plt.title("Evolution of distances")
        plt.legend()
        plt.show() 

    if False:
        label_list = ["Default", "Random","Ef > 0.6", "Ef > 0.7", "d*=2"]
        for i,r_function in enumerate(rejection_function_list):
            cloud_r_opt, R_list, T_list, neighbors_list, distance_list = icp_ultimate(cloud_r, cloud_o, iteration, tol, filename, selecting_function, weighting_function, r_function)
            plt.plot(range(len(distance_list)), distance_list,'k-', label = label_list[i])

        plt.xlabel("Iterations")
        plt.ylabel("Distance")
        plt.title("Evolution of distances")
        plt.legend()
        plt.show() 
