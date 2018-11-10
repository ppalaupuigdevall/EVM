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
    D_intra = pairwise_euclidean_distance(Xl, Xl)
    D_intra = D_intra.numpy()
    sns.distplot(np.resize(D_intra, (np.shape(D_intra)[0]*np.shape(D_intra)[1], 1)), hist = False, kde = True, rug = True,
             color = 'green', 
             kde_kws={'linewidth': 2},
             rug_kws={'color': 'green'})
    
    # Compute distances for inter samples for Cl class
    Xl, Xnotl = select_class(y[i], X, y)
    D_inter = pairwise_euclidean_distance(Xl, Xnotl)
    D_inter = D_inter.numpy()
    plt.title(y[i])
    sns.distplot(np.resize(D_inter, (np.shape(D_inter)[0]*np.shape(D_inter)[1], 1)), hist = False, kde = True, rug = True,
             color = 'red', 
             kde_kws={'linewidth': 2},
             rug_kws={'color': 'red'})
    plt.legend(('Intra (same class)', 'Inter (different classes)'))
    i += 1
    plt.show()
    
