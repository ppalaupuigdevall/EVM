import numpy as np
import face_recognition
import h5py
import torch

weibull_params =  []
sorted_names = []
def load_weibull_params(weibull_file):
    global weibull_params, sorted_names 
    def append_weibull(obj):
        global weibull_params
        weibull_params.append(obj)    
    
    def get_objects(obj, name):
        global sorted_names
        #if("weibull" in name):
        append_weibull(obj)
        sorted_names.append(name.name[1:])
    with h5py.File(weibull_file, 'r') as f:
        f.visititems(get_objects)
    return weibull_params, sorted_names

def load_weibull_params_2(weibull_file):
    global weibull_params, sorted_names
    f = h5py.File(weibull_file, 'r')
    # f.keys sorts the keys we don't have to do sorted()
    sorted_names = list(f.keys())
    for key in sorted_names:
        weibull_params.append(f[key])
    return weibull_params, sorted_names
 
def ppp_cosine_similarity(x1, x2):
    """Computes pairwise cosine similarity between rows of both matrices 
    :param x1: (Nl x dimension_of_feature_vector) PYTORCH tensor containing the feature vectors of each instance of Cl
    :param x2: (Nnotl x dimension_of_feature_vector) PYTORCH tensor containing the feature vectors of each instance of not classes Cl
    :return: (Nl x Nnotl) PYTORCH tensor
    """
    x1_normalized = x1 / x1.norm(dim=1, p = 2)[:, None]
    x2_normalized = x2 / x2.norm(dim=1, p = 2)[:, None]
    res = torch.mm(x1_normalized, x2_normalized.t_())
    return res

def pairwise_euclidean_distance(x, y):
    """Computes pairwise euclidean distance between two matrices.
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

def psi_i_dist(dist, lambda_i, k_i):
    """Gives the probability of sample inclusion
    :param dist: Numpy vector of distances between samples
    :param lambda_i: Scale of the Weibull fitting
    :param k_i: Shape of the Weibull fitting
    :return: PSI = Probability of Sample Inclusion. This is the probability that x' is included in the boundary estimated by x_i
    """
    return np.exp(-(((np.abs(dist))/lambda_i)**k_i))

def face_identification_evt(query_face, enrollment_faces, enrollment_labels, d_thresh, hdf5_file, distance=0):
    # First the 	   
    id = 'Unknown'
    scores = np.zeros(27)
    # This line returns a list containing pointers in the disk where the weibull parameters are located, this is list is properly sorted according to the names of the known_faces
    weibull_params_dataset_list, sorted_names = load_weibull_params_2(hdf5_file)
    
    #print("EL TIPUS DINS DEL SORTED NAME ES: "+str(type(sorted_names[0])))
    #print("La llista amb els weibull: ")
    #print(type(weibull_params_dataset_list[0]))
    # Functions that compute distance require pytorch tensors
    query_face_t = torch.from_numpy(query_face)
    enrollment_faces_t = torch.from_numpy(np.asarray(enrollment_faces)) 
    if(distance==0):
        distances = pairwise_euclidean_distance(query_face_t.view(-1,4096), enrollment_faces_t.view(len(enrollment_faces), 4096))
    else:
        distances = ppp_cosine_similarity(query_face_t.view(-1, 4096), enrollment_faces_t.view(len(enrollment_faces), 4096))
    distances_numpy = distances.numpy()

    for i in range(0, len(sorted_names)):
        weibull_numpy_mat = weibull_params_dataset_list[i]
        weibull_numpy_mat = weibull_numpy_mat[:]
        
        # Load the distances for class i
        distances_belonging_to_that_class = distances_numpy[:, enrollment_labels.index(sorted_names[i])]       
        number_of_weibull_pairs = len(weibull_numpy_mat[:, 0])
        # PSI_for_each weibull is a matrix holding the evaluation of the distance for each pair of weibull params
        psi_for_each_weibull = np.zeros((number_of_weibull_pairs, len(distances_belonging_to_that_class)))
        for j in range(0, number_of_weibull_pairs):
            if(distance==0):
                psi_for_each_weibull[j] = psi_i_dist(distances_belonging_to_that_class, weibull_numpy_mat[j, 0], weibull_numpy_mat[j, 1])
            else:
                #psi_for_each_weibull[j] = 1.0 - psi_i_dist(distances_belonging_to_that_class, weibull_numpy_mat[j, 0], weibull_numpy_mat[j, 1])
                psi_for_each_weibull[j] = 1.0 - psi_i_dist(distances_belonging_to_that_class, weibull_numpy_mat[j, 0], weibull_numpy_mat[j, 1])
        scores[i] = np.min(psi_for_each_weibull)

    # compute largest score
    for i in range(0, len(sorted_names)):
        strong_candidates = np.where(scores >= d_thresh)[0]
        if(strong_candidates.size != 0):
            id = sorted_names[np.argmax(scores)]

    return id, scores