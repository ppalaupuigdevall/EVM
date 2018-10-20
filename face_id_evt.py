import numpy as np
import face_recognition
import h5py


weibull_params =  []
extreme_vectors = []
def load_weibull_params(weibull_file):
    
    def append_EV(obj):
        global extreme_vectors
	extreme_vectors.append(obj)
    
    def append_weibull(obj):
        global weibull_params
	weibull_params.append(obj)    
    
    def get_objects(obj, name):
	if("weibull" in name):
	    append_weibull(obj)
    
    f.visititems(get_objects)
    return extreme_vectors, weibull_params
		
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
    return torch.clamp(dist, 0.0, np.inf)        


def face_identification_evt(query_face, enrollment_faces, enrollment_labels, d_thresh):
    hdf5_file = /work/ppalau/Extreme_Value_Machine/dev_enrolment_weibulls.hdf5
    id = 'Unknown'
    scores = np.zeros(34)
    extreme_vectors_dataset_list, weibull_params_dataset_list = load_weibull_params(hdf5_file)
    
    for face in enrollment_faces:
	face_index = enrollment_labels.index(Cl)
        Xl_hdf5_dataset = extreme_vectors_dataset_list[face_index]
	Weibull_dataset = weibull_params_dataset_list[face_index]
	xl_numpy_matrix = Xl_hdf5_dataset[:]
	weibull_numpy_matrix = Weibull_dataset[:]
	distances = ppp_cosine_similarity(torch.from_numpy(query_face), torch.from_numpy(Xl_numpy_matrix).float())
	Nl = len(weibull_numpy_matrix[:,0])
	distances_numpy = distances.numpy()[0]
	probabilities_class_i = np.zeros(Nl)
	for i in range(0, Nl):
	    probabilities_class_i[i] = 1.0 - psi_i_dist( distances_numpy[i], weibull_numpy_matrix[i, 0], weibull_numpy_matrix[i, 1])
	scores[face_index] = np.min(probabilities_class_i)    

    # compute largest score
    for i in range(0, len(enrollment_labels)):
	strong_candidates = np.where(scores >= d_thresh)[0]
	if(strong_candidates.size != 0):
	    id = enrollment_labels[np.argmax(scores)]
    return id, scores
