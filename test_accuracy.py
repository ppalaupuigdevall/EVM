from face_id_evt import *
import os

def load_data_from_folders_with_labels(root_dir):
    llis = []
    known_classes = []
    labels = []
    for root, dirs, files in os.walk(root_dir):
        print("Ordre en que carreguem dels folders")
        dirs = sorted(dirs)
        print(dirs)
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

test_dir = '/work/ppalau/Extreme_Value_Machine/feature_vectors_imagenet/alexnet/euclidean/test/known_classes/'

llis, known_classes_, labels_ = load_data_from_folders_with_labels('/work/ppalau/Extreme_Value_Machine/feature_vectors_imagenet/alexnet/euclidean/train/known_classes/')

#wei, sorted_classes = load_weibull_params_2('/work/ppalau/Extreme_Value_Machine/alexnet_imagenet_200_0_mr_fit_low.hdf5')

#test_feat = np.load('/work/ppalau/Extreme_Value_Machine/feature_vectors_imagenet/alexnet/euclidean/test/known_classes/n01491361/n01491361_.JPEG.npy')
#test_feat = np.load('/work/ppalau/Extreme_Value_Machine/feature_vectors_imagenet/alexnet/euclidean/test/known_classes/unknown_classes_1/n13037406_9917.JPEG.npy')
#id_, scores_ = face_identification_evt(test_feat, llis, labels_, 1e-50, '/work/ppalau/Extreme_Value_Machine/alexnet_imagenet_200_1_fit_.hdf5', 1)
wei, sorted_classes = load_weibull_params_2('/work/ppalau/Extreme_Value_Machine/alexnet_imagenet_200_0_mr_fit_low.hdf5')
"""Iterate over all test classes, that is, known + unknown"""
correct = 0
total = 0
print("Classes in test dir:")
print(os.listdir(test_dir))
for class_ in sorted(os.listdir(test_dir)):
    print(class_)

    for file_ in os.listdir(os.path.join(test_dir, class_)):
        test_feat = np.load(os.path.join(test_dir, class_, file_))
        id_, scores_ = face_identification_evt(test_feat, llis, labels_, 3.5e-3, '/work/ppalau/Extreme_Value_Machine/alexnet_imagenet_200_1_fit_.hdf5', 1)
        print("Class: " + str(class_))
        print("Predicted: " +str(id_))
        
        if(id_ == class_):
            print("correct")
            correct+=1
        else:
            print("incorrect")
        total+=1

accuracy = correct/total
print("The accuracy is = " + str(accuracy))