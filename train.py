import numpy as np
import h5py
import torch
import argparse
import scipy
import os
import libmr

#dimension of the feature vectors extracted in feature_extractor.py
dimension = (1, 128)


def psi_i_dist(dist, lambda_i, k_i):
    """
    Gives the probability of sample inclusion
    :param dist: Numpy vector of distances between samples
    :param lambda_i: Scale of the Weibull fitting
    :param k_i: Shape of the Weibull fitting
    :return: PSI = Probability of Sample Inclusion. This is the probability that x' is included in the boundary estimated by x_i
    """
    return np.exp(-(((np.abs(dist))/lambda_i)**k_i))


def fit_(x, iters=100, eps=1e-6):
    """THIS FUNCTION IS TAKEN FROM https://github.com/mlosch/python-weibullfit/blob/master/weibull/backend_numpy.py"""
    """
    Fits a 2-parameter Weibull distribution to the given data using maximum-likelihood estimation.
    :param x: 1d-ndarray of samples from an (unknown) distribution. Each value must satisfy x > 0.
    :param iters: Maximum number of iterations
    :param eps: Stopping criterion. Fit is stopped ff the change within two iterations is smaller than eps.
    :return: Tuple (Shape, Scale) which can be (NaN, NaN) if a fit is impossible.
        Impossible fits may be due to 0-values in x.
    """
    # fit k via MLE
    ln_x = np.log(x)
    k = 1.
    k_t_1 = k

    for t in range(iters):
        x_k = x ** k
        x_k_ln_x = x_k * ln_x
        ff = np.sum(x_k_ln_x)
        fg = np.sum(x_k)
        f = ff / fg - np.mean(ln_x) - (1. / k)

        # Calculate second derivative d^2f/dk^2
        ff_prime = np.sum(x_k_ln_x * ln_x)
        fg_prime = ff
        f_prime = (ff_prime/fg - (ff/fg * fg_prime/fg)) + (1. / (k*k))

        # Newton-Raphson method k = k - f(k;x)/f'(k;x)
        k -= f/f_prime

        if np.isnan(f):
            return np.nan, np.nan
        if abs(k - k_t_1) < eps:
            break

        k_t_1 = k

    lam = np.mean(x ** k) ** (1.0 / k)

    return k, lam


def reduce(PSI_l, Xl, coverage_threshold):
    """
    Computes set cover reduction to get the most relevant samples that define the class Xl.
    :param PSI_l: (Nl x 2) matrix containing both the scale and the shape of the weibull distribution
    :param Xl: (Nl x dimension_feature_vector) matrix containing the feature vectors of each instance of a class
    :param coverage_threshold: Probability above which we consider an instance to be not enough representative of its class
    :return: The indexes of the most representative samples of a class
    """
    #This matrix D is symmetric
    D = ppp_cosine_similarity(Xl, Xl)
    # Number of instances of the class
    Nl = np.shape(D)[0]

    S = []
    for i in range(Nl):
        Si = []
        for j in range(Nl):
            if(psi_i_dist(D[i, j], PSI_l[i,0], PSI_l[i, 1]/10) >= coverage_threshold):
                # Sample i is redundant with respect to j
                Si.append(j)
        S.append(Si)
    # Universe
    U = list(range(0, Nl))
    # Covered index
    C = []
    # Final indexs
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


def ppp_cosine_similarity(x1, x2):
    """
    Computes pairwise cosine similarity between rows of both matrices
    :param x1: (Nl x dimension_of_feature_vector) PYTORCH tensor containing the feature vectors of each instance of Cl
    :param x2: (Nnotl x dimension_of_feature_vector) PYTORCH tensor containing the feature vectors of each instance of not classes Cl
    :return: (Nl x Nnotl) PYTORCH tensor
    """
    x1_normalized = x1 / x1.norm(dim=1, p = 2)[:, None]
    x2_normalized = x2 / x2.norm(dim=1, p = 2)[:, None]
    res = torch.mm(x1_normalized, x2_normalized.t_())
    return res


def pairwise_euclidean_distance(x, y):
    """
    Computes pairwise euclidean distance between two matrices.
    :param x: (Nl x dimension_of_feature_vector) PYTORCH tensor containing the feature vectors of each instance of Cl
    :param y: (Nnotl x dimension_of_feature_vector) PYTORCH tensor containing the feature vectors of each instance of not classes Cl
    :return: (Nl x Nnotl) PYTORCH tensor
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.sqrt(torch.clamp(dist, 0.0, np.inf))


def select_class(Cl, X, y):
    """
    Selects class Cl and not class Cl from the list X
    :param Cl: Class identifier from list known_classes
    :param X: List containing matrices of (Nl x dimension_of_feature_vector) of all training classes
    :return: Tuple: (Xl, Xnotl) : Xl --> (Nl x dimension_of_feature_vector) PYTORCH tensor containing the feature vectors of each instance of Cl
                                : Xnotl -->  (Nnotl x dimension_of_feature_vector) PYTORCH tensor containing the feature_vectors of each instance of not classes Cl
    """
    Xnotl = []
    Xl = X[y.index(Cl)]
    Xnotl_ind = [i for i, c in enumerate(y) if (Cl != c)]
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

def select_class_list(Cl, X, y):
    Xnotl = []
    Xl = X[y.index(Cl)]
    Xnotl_ind = [i for i, c in enumerate(y) if (Cl != c)]
    Nnotl = 0
    for ind in Xnotl_ind:
        Nnotl += len(X[ind][:,0])
        Xnotl.append(X[ind])
    return Xl, Xnotl
	
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
    """
    Returns the Weibull parameters of each instance of class Cl, that is, the parameters that model the
    distribution of the shortest 'tailsize' margins of that class with respect to all other classes
    :param X: List containing matrices of (Nl x dimension_of_feature_vector) of all training classes
    :param y: Labels of the classes
    :param tailsize: Number of margins to be fitted by the Weibull distribution
    :param Cl: Class identifier from list known_classes
    :return: PSI_l --> (Nl x 2) matrix containing the scale (lambda) and shape (k) of the fitted margins for each instance
                    of the class l
    """
    Xl, Xnotl = select_class(Cl, X, y)
    # distance computation
    D = ppp_cosine_similarity(Xl, Xnotl)
    #D = pairwise_euclidean_distance(Xl, Xnotl)
    D = D.numpy() 
    #print(D)
    Nl = len(Xl[:, 0])
    # PSI_l is formed by (lambda, k)
    PSI_l = np.zeros((Nl, 2))
    #mr = libmr.MR()
    for i in range(0, Nl):
        # We want to know the distribution of the MARGINS (we have to divide by 2 because the margin is the point that is half-way the negative sample)
        # We have to sort the vector of distances because we are interested in the closest instances, that are the most important defining the margins
        # because they can create confusion. NOTE = 0.5 is because is a margin
        d_sorted = 0.5 * np.sort(D[i, :])[:tailsize]
        k_i, lambda_i = fit_(d_sorted, iters = 100, eps = 1e-6)
        #mr.fit_high(d_sorted, tailsize)
        PSI_li = (lambda_i, k_i)
        #PSI_li = mr.get_params()[:2]
        PSI_l[i, :] = PSI_li
    return PSI_l


def train_EVM(X, y, tailsize, coverage_threshold):
    """

    :param X:
    :param y:
    :param tailsize:
    :param coverage_threshold:
    :return:
    """
    global output_file
    with h5py.File(output_file, 'w') as fi:
        for Cl in y:
            print(Cl)
            PSI_l = fit(X, y, tailsize, Cl)
            print("La mitjana del parametre k es = " +str(np.mean(PSI_l[:,1])))
            Xl, Xnotl = select_class(Cl, X, y)
            # TO BE REVIEWED
            #################################
            # I = reduce(PSI_l, Xl, coverage_threshold)
            # Xl_reduced = Xl[I, :]
            # PSI_l_reduced = PSI_l[I, :]
            # print("Original shape of the PSI matrix for the class " + str(Cl) + " = " + str(np.shape(PSI_l)))
            # print("Reduced shape of the PSI matrix for the class " + str(Cl) + " = " + str(np.shape(PSI_l_reduced)))
            #################################
            #fg = fi.create_group(Cl)
            data_weibull = fi.create_dataset(Cl, np.shape(PSI_l), dtype ='f4', data = PSI_l)
            #reduced_extreme_vectors = fg.create_dataset(Cl + "_extreme_vectors", np.shape(Xl), dtype='f4', data = Xl)


llista = []
def load_data_from_HDF5(f):
    """
    Function to load Feature vectors from each sample of a class, the data structure of the hdf file is explained in feature_extractor.py
    :param f: hdf file where feature vectors are located
    :return: list containing the h5py datasets (sets of feature vectors) for each training class
    """
    # Auxiliary functions defined to load the data
    def myappend(obj):
        global llista
        llista.append(obj)
    def get_objects(name, obj):
        if(("Train" in name) and ("Conegudes" in name)):
            myappend(obj)

    f.visititems(get_objects)
    return llista

def load_data_from_folders(root_dir):
    llis = []
    known_classes = []
    for root, dirs, files in os.walk(root_dir):
        for dir in dirs:
            X = np.zeros((1,128))
            for fil in os.listdir(os.path.join(root_dir,dir)):
                a = np.load(os.path.join(root_dir,dir,fil))
                a = a.T
                X = np.vstack((X,a))
            # Remove the first row of X
            X = X[1:, :]
            llis.append(X)
            known_classes.append(dir)
        break
    return llis, known_classes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("rootdir", help = "Directory where the folders containing feature vectors are located", type=str)
    parser.add_argument("tailsize", help = "Number of margins to fit the weibull distribution", type = int)
    parser.add_argument("outdir", help = "The output file where the hdf5 file with Weibull parameters for each class will be located, sth like = /mydir/weibulls.hdf5", type = str)
    args = parser.parse_args()
    rootdir = args.rootdir
    tailsize = args.tailsize
    output_file = args.outdir
    coverage_threshold = 0.5
    
    X, y = load_data_from_folders(rootdir)
    
    print(y)
    train_EVM(X, y, tailsize, coverage_threshold)
    # with h5py.File(r"C:\Users\user\Pon\MET\IR\Datasets\Imagenet_Ponc\ALEXNET_imagenetponc_feature_vectors_PROPERLY_SELECTED_2.hdf5", 'r') as f:
    #     X = load_data_from_HDF5(f)
    #     train_EVM(X, y, tailsize, coverage_threshold)
