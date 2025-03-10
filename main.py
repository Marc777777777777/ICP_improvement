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