import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import h5py
import numpy as np
from train import ppp_cosine_similarity
from train import select_class
from train import psi_i_dist

__author__ = "Ponc Palau Puigdevall"
# Labels
known_classes = ('n02096294','n03733805','n02123394')
unknown_classes = ('n02109047', 'n02110185')
#known_classes = ('n02096294','n02110185','n02119634','n02123394', 'n02504458')
dimension = (1, 4096)



def get_feature_vector(img, layer, dimension):
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    # 3. Create a vector of zeros that will hold our feature vector
    #    The fully connected layer has a size of 4096
    my_embedding = torch.zeros(dimension)
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector AND THE NUMBER OF CHANNELS OF THE IMAGE IN CASE THERE IS ONE GRAY IMAGE IN THE DATASET(WE HAVE TO DISCARD IT)
    return my_embedding

extreme_vectors = []
weibull_params = []

def load_data_from_HDF5_infer(f):
    # Auxiliary functions defined to load the data
    def myappend_EV(obj):
        global extreme_vectors
        extreme_vectors.append(obj)
    def my_append_Weibull(obj):
        global weibull_params
        weibull_params.append(obj)
    def get_objects(name, obj):
        if(("_extreme_vectors" in name)):
            myappend_EV(obj)
        elif("_weibull" in name):
            my_append_Weibull(obj)

    f.visititems(get_objects)
    return extreme_vectors, weibull_params
    #####################################

known = True
def load_test_feature_vectors(fi ,class_name):
    def get_objects_test(name, obj):
        if((class_name in name) and ("Test" in name)):
            return obj
    ds = fi.visititems(get_objects_test)
    return ds


#Hyperparameter

probability_threshold = 0.5

image_name = r"C:\Users\user\Ponç\MET\IR\Datasets\Imagenet_Ponc\Uncompressed\Conegudes\n02123394\Test\n02123394_284.JPEG"

#Load a pretrained model of AlexNet architecture
model = models.alexnet(pretrained = True)
#Extract the last fully connected layer fc7[4096]
layer = model.classifier[5]
# Set the model to evaluation mode, no dropout is applied during forward pass
model.eval()
#dimension of the desired layer
dimension = (1, 4096)

# The model accepts images of size 224x224
scaler = transforms.Resize((224, 224))
# The model accepts data in a [0,1] range. Then we have to normalize it according to the following rule.
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
# This line converts the PIL image to a pytorch tensor
to_tensor = transforms.ToTensor()

img = Image.open(image_name)

#feature_vect = get_feature_vector(img, layer, dimension)

with h5py.File(r"C:\Users\user\Ponç\MET\IR\Datasets\Imagenet_Ponc\ALEXNET_imagenetponc_feature_vectors_PROPERLY_SELECTED_2.hdf5", 'r') as fi:
    pred = unknown_classes[0]
    test_dataset = load_test_feature_vectors(fi, pred)
    test_dataset_mat = test_dataset[:]
    feature_vect_nu = test_dataset_mat[np.random.randint(0, len(test_dataset_mat[:,0])), :]
    print("The test image, which the EVM has never seen, is from class = " + str())
    feature_vect = torch.from_numpy(feature_vect_nu).float().view(-1, 4096)
    print(feature_vect.size())
with h5py.File(r"C:\Users\user\Ponç\MET\IR\Datasets\Imagenet_Ponc\ALEXNET_imagenetponc_trained_weibull_parameters.hdf5", 'r') as fi:
    # EVs is a list containing the extreme vectors of each training class, theoretically these extreme vectors
    # summarize each class because model reduction has been applied
    EVs, Weibulls = load_data_from_HDF5_infer(fi)
    probabilities = []
    for Cl in known_classes:
        Xl_hdf5_dataset = EVs[known_classes.index(Cl)]
        Weibulls_dataset = Weibulls[known_classes.index(Cl)]
        Xl_numpy_matrix = Xl_hdf5_dataset[:]
        Weibulls_numpy_matrix = Weibulls_dataset[:]
        distances = (ppp_cosine_similarity(feature_vect, torch.from_numpy(Xl_numpy_matrix).float()))
        # Aquest 1.0 - és perque com es cosine similarity que dos vectors s'assemblin molt vol dir 0
        probabilities.append(1.0 - np.max(psi_i_dist(distances.numpy(), Weibulls_numpy_matrix[0, 0], Weibulls_numpy_matrix[0, 1])))
    print(known_classes)
    print(probabilities)
