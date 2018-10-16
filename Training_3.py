import h5py
import libmr
import torch.nn as nn
import time
import torch
import numpy as np
import scipy

# Labels
known_classes = ('n02096294','n02110185','n02119634','n02123394', 'n02504458')
dimension = (1, 4096)

# dist will be a vector of distances
def psi_i_dist(dist, lambda_i, k_i):
    return np.exp(-(((np.abs(dist))/lambda_i)**k_i))

def reduce(PSI_l, Xl, coverage_threshold):
    #This matrix D is symmetric
    D = ppp_cosine_similarity(Xl, Xl)
    # Number of instances of the class
    Nl = np.shape(D)[0]
    #distance_probabilities = np.zeros((np.shape(D)))

    print(D[0,:])
    S = []
    for i in range(Nl):
        Si = []
        for j in range(Nl):
            print(psi_i_dist(D[i, j], PSI_l[i,0], PSI_l[i, 1]))
            if(psi_i_dist(D[i, j], PSI_l[i,0], PSI_l[i, 1]) >= coverage_threshold):
                # Si l'afegim vol dir que j és redundant respecte i (aquesta i amb aquella j es solapen més del 50%)
                Si.append(j)
                print(psi_i_dist(D[i, j], PSI_l[i, 0], PSI_l[i, 1]))
        S.append(Si)

    U = list(range(0, Nl))
    C = []
    I = []
    #Set Cover Implementation
    while (len(scipy.intersect1d(C, U)) != len(U)):
        # punct_ref is a counter to find the maximum in every iteration
        punct_ref = 0
        # ind represent the index that we will append to our index's list
        ind = 0
        index_s = 0
        for s in S:
            punct = 0
            relative_inclusion = scipy.isin(s, C)
            for eleme in relative_inclusion:
                if (eleme is False):
                    punct += 1
            if (punct >= punct_ref):
                ind = index_s
            index_s += 1

        C = scipy.union1d(C, S[ind])
        I.append(ind)
        S.remove(S[ind])
        if (len(S) == 0):
            break
    return I

# x1 and x2 are two tensors one containing feature vectors of class l and the other not l
def ppp_cosine_similarity(x1, x2):
    x1_normalized = x1 / x1.norm(dim=1, p = 2)[:, None]
    x2_normalized = x2 / x2.norm(dim=1, p = 2)[:, None]
    res = torch.mm(x1_normalized, x2_normalized.t_())
    return res

"""
FUNCTION: select_class(Cl, X):
Description: Selects class Cl and not class Cl from the list X
Input Parameters:
        X       --> List containing matrices of (Nl x dimension_of_feature_vector) of all training classes
        Cl      --> Class identifier from list known_classes
Output parameters:
        Xl       --> (Nl x dimension_of_feature_vector) matrix containing the feature vectors of each instance of Cl
        Xnotl    --> list containing matrices with the feature_vectors of each instance of not class Cl
"""
def select_class(Cl, X):
    Xnotl = []
    Xl = X[known_classes.index(Cl)]
    Xnotl_ind = [i for i, c in enumerate(known_classes) if (Cl != c)]
    Nnotl = 0
    for ind in Xnotl_ind:
        Nnotl += len(X[ind][:,0])
        Xnotl.append(X[ind])
    # Stack Xnotl matrices vertically
    Xnotl_mat = np.zeros((1, dimension[1]))
    for Xnot_elem in Xnotl:
        Xnotl_mat = np.vstack((Xnotl_mat, Xnot_elem[:]))
    Xnotl_mat = Xnotl_mat[1:Nnotl,:]
    #These are numpy arrays we have to convert them to tensors
    Xl_tensor = torch.from_numpy(Xl[:]).float()
    Xnotl_tensor = torch.from_numpy(Xnotl_mat).float()
    return Xl_tensor, Xnotl_tensor
    #####################################

"""
FUNCTION: fit(X, y, tailsize, Cl):
Description: Returns the Weibull parameters of each instance of class Cl, that is, the parameters that model the 
             distribution of the shortest 'tailsize' margins of that class with respect to all other classes
Input Parameters:
        X       --> List containing matrices of (Nl x dimension_of_feature_vector) of all training classes
        y       --> Labels of the classes
        tailsize--> Number of margins to be fitted by the Weibull distribution
        Cl      --> Class identifier from list known_classes
Output parameters:
        PSI_l   --> (Nl x 2) matrix containing the scale (lambda) and shape (k) of the fitted margins for each instance 
                    of the class l 
"""
def fit(X, y, tailsize, Cl):
    # If other distance wants to be computed, replace it below
    distance = nn.CosineSimilarity(dim=1, eps=1e-6)
    Xl, Xnotl = select_class(Cl, X)
    D = ppp_cosine_similarity(Xl, Xnotl)
    Nl = len(Xl[:, 0])
    # PSI_l = (lambda, k)
    PSI_l = np.zeros((Nl, 2))
    mr = libmr.MR()
    print("The distance matrix is = ")
    print(D)
    for i in range(0, Nl):
        # We want to know the distribution of the MARGINS (we have to divide by 2 because the margin is the point that is half-way the negative sample)
        # We have to sort the vector of distances because we are interested in the closest instances, that are the most important defining the margins
        # because they can create confusion. NOTE = 0.5 is because is a margin
        # COMENTAR AMB RAMON! ==> S'ha de fer un fit_high perque estem utilitzant la cosine similarity no?
        #d_sorted = 0.5 * np.flip(np.sort(D[i, :].numpy()), axis=0)[:tailsize]
        d_sorted = np.sort(1.0 - D[i, :].numpy())[:tailsize]
        #d_sorted = D[i, :].numpy()
        mr.fit_low(d_sorted, tailsize)
        PSI_li = mr.get_params()[:2]
        print(PSI_li)
        PSI_l[i, :] = PSI_li
    return PSI_l
    #####################################

def train_EVM(X, y, tailsize, coverage_threshold):
    with h5py.File(r"C:\Users\user\Ponç\MET\IR\Datasets\Imagenet_Ponc\ALEXNET_imagenetponc_trained_weibull_parameters.hdf5", 'w') as fi:
        for Cl in y:
            print(Cl)
            PSI_l = fit(X, y, tailsize, Cl)
            print("La mitjana del parametre k és = " +str(np.mean(PSI_l[:,1])))
            #Xl, Xnotl = select_class(Cl, X)
            #I = reduce(PSI_l, Xl, coverage_threshold)
            #Xl_reduced = Xl[I, :]
            #PSI_l_reduced = PSI_l[I, :]
            #print("Original shape of the PSI matrix for the class " + str(Cl) + " = " + str(np.shape(PSI_l)))
            #print("Reduced shape of the PSI matrix for the class " + str(Cl) + " = " + str(np.shape(PSI_l_reduced)))
            #fg = fi.create_group(Cl)
            #data_weibull = fg.create_dataset(Cl + "_weibull", np.shape(PSI_l_reduced), dtype ='f4', data = PSI_l_reduced)
            #reduced_extreme_vectors = fg.create_dataset(Cl + "_extreme_vectors", np.shape(Xl_reduced), dtype='f4', data = Xl_reduced)
        #####################################

llista = []
def load_data_from_HDF5(f):
    # Auxiliary functions defined to load the data
    def myappend(obj):
        global llista
        llista.append(obj)
    def get_objects(name, obj):
        if(("Train" in name) and ("Conegudes" in name)):
            myappend(obj)

    f.visititems(get_objects)
    return llista
    #####################################

if __name__ == '__main__':
    tailsize = 4000
    coverage_threshold = 0.99999
    y = known_classes
    with h5py.File(r"C:\Users\user\Ponç\MET\IR\Datasets\Imagenet_Ponc\ALEXNET_imagenetponc_feature_vectors_more_training_classes.hdf5", 'r') as f:
        X = load_data_from_HDF5(f)
        train_EVM(X, y, tailsize, coverage_threshold)