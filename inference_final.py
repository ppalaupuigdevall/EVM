from face_id_evt import *
import os

def load_data_from_folders_with_labels(root_dir):
    llis = []
    known_classes = []
    labels = []
    for root, dirs, files in os.walk(root_dir):
        
        dirs = sorted(dirs)
        
        for dir in dirs:
            X = np.zeros((1, 4096))
            for fil in sorted(os.listdir(os.path.join(root_dir,dir))):
                #print(os.path.join(root_dir,dir,fil))
                a = np.load(os.path.join(root_dir,dir,fil))
                #a = a.T
                known_classes.append(dir)
                llis.append(a)
                #labels.append(dirs.index(dir)) # amb aixo donava valueerror
                labels.append(dir)
        break
    return llis, known_classes, labels

llis, known_classes_, labels_ = load_data_from_folders_with_labels('/work/ppalau/Extreme_Value_Machine/feature_vectors_imagenet/alexnet/euclidean/train/known_classes/')

wei, sorted_classes = load_weibull_params_2('/work/ppalau/Extreme_Value_Machine/alexnet_imagenet_200_0_mr_fit_low.hdf5')
#wei, sorted_classes = load_weibull_params_2('/work/ppalau/Extreme_Value_Machine/alexnet_imagenet_200_0_fit_high.hdf5')
test_feat = np.load('/work/ppalau/Extreme_Value_Machine/feature_vectors_imagenet/alexnet/euclidean/test/known_classes/n01491361/n01491361_9959.JPEG.npy')
#test_feat = np.load('/work/ppalau/Extreme_Value_Machine/feature_vectors_imagenet/alexnet/euclidean/test/unknown_classes_1/n13037406_9917.JPEG.npy')
test_feat = np.load('/work/ppalau/Extreme_Value_Machine/feature_vectors_imagenet/alexnet/euclidean/test/known_classes/n03998194/n03998194_7158.JPEG.npy')
test_feat = np.load('/work/ppalau/Extreme_Value_Machine/feature_vectors_imagenet/alexnet/euclidean/test/known_classes/n01491361/n01491361_10000.JPEG.npy')

id_, scores_ = face_identification_evt(test_feat, llis, labels_, 1e-6, '/work/ppalau/Extreme_Value_Machine/alexnet_imagenet_200_1_fit_.hdf5', 1)
print("Test class: " + str('n01491361'))
print("Predicted class: " + str(id_))
print("Scores for each class: ")
print(scores_)
