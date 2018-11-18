import numpy as np
import matplotlib.pyplot as plt
import torch
from train_EVM import *
import os
import seaborn as sns
number_of_feature_vectors = 34
X = []

for feature_vector_matrix in os.listdir('feature_vectors/'):
    X.append(np.load(os.path.join('feature_vectors',feature_vector_matrix)))
    print(feature_vector_matrix)
y = ['Carles_Puigdemont', 'Pablo_Iglesias', 'Pedro_Sanchez', 'Javier_Fernandez', 'Sergio_Martin', 'Alfredo_Perez_Rubalcaba', 'Ignacio_Aguado', 'Angel_Gabilondo', 'Angela_Merkel', 'Cesar_Luena', 'Patxi_Lopez', 'Begona_Villacis', 'Moises_Rodriguez', 'Jose_Hervas', 'Enrique_Dans', 'Mariano_Rajoy', 'David_Cameron', 'Antonio_Perez_Henares', 'Albert_Rivera', 'Papa_Francisco', 'Ricardo_Martin', 'Alberto_Garzon', 'Carmelo_Encinas', 'Esther_Esteban', 'Felipe_VI', 'Francois_Hollande', 'Jose_Maria_Villegas', 'Maria_Gonzalez', 'Carlos_Chaguaceda', 'Julio_Somoano', 'Meritxell_Batet', 'Olga_Lambea', 'Papa_Juan_XXIII', 'Vladimir_Putin']
y = list(np.sort(y))

i = 0
for Xl in X:
    # Compute distances for intra samples for Cl class
    Xl = torch.from_numpy(Xl).float()
    D_intra_eucl = pairwise_euclidean_distance(Xl, Xl)
    D_intra_eucl = D_intra_eucl.numpy()
    D_intra_cos = ppp_cosine_similarity(Xl, Xl).numpy()
    


    
    
    # Compute distances for inter samples for Cl class
    Xl, Xnotl = select_class(y[i], X, y)
    D_inter_eucl = pairwise_euclidean_distance(Xl, Xnotl)
    D_inter_eucl = D_inter_eucl.numpy()
    D_inter_cos = ppp_cosine_similarity(Xl, Xnotl).numpy()
    Nl = np.shape(D_inter_eucl)[0]
    PSI_l = np.zeros((Nl, 2))
    for j in range(0, Nl):
        # We want to know the distribution of the MARGINS (we have to divide by 2 because the margin is the point that is half-way the negative sample)
        # We have to sort the vector of distances because we are interested in the closest instances, that are the most important defining the margins
        # because they can create confusion. NOTE = 0.5 is because is a margin
        d_sorted = 0.5 * np.sort(D_inter_eucl[j, :])[:300]
        k_i, lambda_i = fit_(d_sorted, iters = 100, eps = 1e-6)
        #mr.fit_high(d_sorted, tailsize)
        PSI_li = (lambda_i, k_i)
        #PSI_li = mr.get_params()[:2]
        PSI_l[j, :] = PSI_li
    print("Euclidean = ")
    print("k_i, lambda_i = " + str(np.mean(PSI_l[:,0])) + ', ' + str(np.mean(PSI_l[:,1])))
    

    for j in range(0, Nl):
        # We want to know the distribution of the MARGINS (we have to divide by 2 because the margin is the point that is half-way the negative sample)
        # We have to sort the vector of distances because we are interested in the closest instances, that are the most important defining the margins
        # because they can create confusion. NOTE = 0.5 is because is a margin
        d_sorted = 0.5 * np.sort(D_inter_cos[j, :])[:300]
        k_i, lambda_i = fit_(d_sorted, iters = 100, eps = 1e-6)
        #mr.fit_high(d_sorted, tailsize)
        PSI_li = (lambda_i, k_i)
        #PSI_li = mr.get_params()[:2]
        PSI_l[j, :] = PSI_li
    print("Cosine = ")
    print("k_i, lambda_i = " + str(np.mean(PSI_l[:,0])) + ', ' + str(np.mean(PSI_l[:,1])))
    
    # plt.subplot(121)
    # plt.title(y[i])
    # sns.distplot(np.resize(D_intra_eucl, (np.shape(D_intra_eucl)[0]*np.shape(D_intra_eucl)[1], 1)), hist = False, kde = True, rug = True,
    #          color = 'green', 
    #          kde_kws = {'linewidth': 2},
    #          rug_kws = {'color': 'green'})
    # sns.distplot(np.resize(D_inter_eucl, (np.shape(D_inter_eucl)[0]*np.shape(D_inter_eucl)[1], 1)), hist = False, kde = True, rug = True,
    #          color = 'red', 
    #          kde_kws={'linewidth': 2},
    #          rug_kws={'color': 'red'})
    # plt.legend(('Intra (same class)', 'Inter (different classes)'))
    i += 1
    #print("Maximum intra = " + str(np.max(D_intra_eucl)))
    #print("Maximum inter = " + str(np.max(D_inter_eucl)))


    # plt.subplot(122)
    # sns.distplot(np.resize(D_intra_cos, (np.shape(D_intra_cos)[0]*np.shape(D_intra_cos)[1], 1)), hist = False, kde = True, rug = True,
    #          color = 'green', 
    #          kde_kws = {'linewidth': 2},
    #          rug_kws = {'color': 'green'})
    # sns.distplot(np.resize(D_inter_cos, (np.shape(D_inter_cos)[0]*np.shape(D_inter_cos)[1], 1)), hist = False, kde = True, rug = True,
    #          color = 'red', 
    #          kde_kws = {'linewidth': 2},
    #          rug_kws = {'color': 'red'})
    # plt.show()


"""Now I want to draw the k_i and lambda_i distributions for the D_inter_eucl cosine similarity and for the euclidean distance"""