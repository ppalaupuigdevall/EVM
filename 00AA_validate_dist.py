import torch 
import numpy as np

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


x0 = np.array([-2.34733194e-01,  7.89686888e-02,  9.01605338e-02, -6.47947639e-02,
       -7.39788637e-02,  7.65455142e-03, -6.68892562e-02, -3.86098027e-03,
        7.65804648e-02, -1.20447367e-01,  1.87569171e-01, -6.48702085e-02,
       -1.90515235e-01, -7.62166083e-02,  3.76493186e-02,  1.03946403e-01,
       -1.40077502e-01, -1.30167782e-01, -1.29645303e-01, -1.62752151e-01,
        8.54563788e-02,  1.03243843e-01, -1.34515405e-01, -6.09022379e-03,
       -2.70824254e-01, -3.17694187e-01, -1.08711824e-01, -4.46606874e-02,
        4.05016616e-02, -1.23013906e-01,  3.09771579e-02, -6.02088943e-02,
       -1.91043600e-01, -1.35344788e-02,  4.09381986e-02,  1.55613035e-01,
       -7.58131370e-02, -6.43141493e-02,  1.41499221e-01, -7.10371044e-03,
       -1.14297263e-01,  8.36025178e-02,  6.48667067e-02,  2.99117416e-01,
        1.38284609e-01,  4.10387814e-02, -1.58391967e-02, -7.92214796e-02,
        1.53625801e-01, -2.52203435e-01,  1.13785028e-01,  1.75477356e-01,
        1.47904828e-02,  1.30402505e-01,  1.94109753e-01, -1.07485116e-01,
       -3.65722924e-03,  1.89570069e-01, -1.64189547e-01,  4.25686911e-02,
       -2.46994104e-03, -5.36957756e-02, -3.22643965e-02, -1.13244131e-01,
        1.41758367e-01,  1.22112721e-01, -7.72307813e-02, -1.68354690e-01,
        2.30725154e-01, -1.08847976e-01, -3.22457962e-02,  8.06790069e-02,
       -4.57589775e-02, -1.37532070e-01, -2.69503593e-01,  1.13747612e-01,
        3.83472234e-01,  1.45788118e-01, -1.95655525e-01,  5.08078635e-02,
       -1.96811818e-02, -4.85087447e-02,  1.40915453e-01, -7.68484920e-03,
       -7.24054873e-02, -2.23535560e-02, -3.38422358e-02,  1.39753699e-01,
        2.52236307e-01,  5.39094657e-02,  1.91568881e-02,  1.73357069e-01,
        8.16375911e-02, -1.97776686e-02,  1.12503424e-01, -2.18640566e-02,
       -1.32621139e-01, -8.71063471e-02, -1.16930813e-01, -9.44858789e-03,
        9.93958563e-02, -9.03223157e-02, -8.12395662e-03,  1.27136976e-01,
       -1.31495029e-01,  2.47584134e-01,  7.25196898e-02, -5.68555370e-02,
       -9.09584537e-02,  2.32680216e-02, -1.24116085e-01,  7.87393302e-02,
        1.49452448e-01, -2.81970769e-01,  2.29885727e-01,  1.15095094e-01,
        3.31446528e-04,  1.58172473e-01,  8.83922279e-02,  2.39568353e-02,
       -4.09523621e-02, -2.77368054e-02, -1.70732409e-01, -9.69994962e-02,
        3.73513773e-02, -2.28425860e-03,  5.96714169e-02,  4.77716997e-02])
x1 = np.array([-0.16545474,  0.0559395 ,  0.06945679, -0.05717231, -0.06817253,
        0.05645578, -0.01422401, -0.06236646,  0.07969561, -0.16774316,
        0.21009181, -0.06164182, -0.2220093 , -0.10458644,  0.02582187,
        0.12509803, -0.15173412, -0.13342282, -0.11188283, -0.1305444 ,
        0.02530028,  0.07170689, -0.09446984, -0.00786146, -0.22472877,
       -0.28643508, -0.08377917, -0.0750376 ,  0.06517574, -0.11041751,
        0.0249692 ,  0.00533317, -0.17588752,  0.00100252,  0.01102671,
        0.13740418, -0.08005667, -0.08678734,  0.17826889,  0.00863225,
       -0.14058397,  0.03818486,  0.06448367,  0.2780799 ,  0.16837694,
       -0.02122522, -0.00387389, -0.05039152,  0.13090077, -0.31943524,
        0.07861835,  0.12220631,  0.03732535,  0.14790097,  0.13955549,
       -0.15354055,  0.01440063,  0.16040089, -0.18110648,  0.05828384,
       -0.00110684, -0.08436507, -0.00864549, -0.03347031,  0.15473514,
        0.10814563, -0.08139429, -0.16289741,  0.21507783, -0.15560268,
       -0.05575593,  0.10088915, -0.04872895, -0.16597591, -0.3037633 ,
        0.11517033,  0.36849519,  0.22216818, -0.15460335,  0.05562546,
       -0.01442445, -0.04661584,  0.14682231,  0.05486565, -0.02756013,
       -0.02829467, -0.02382531,  0.09918835,  0.22147116,  0.03294865,
       -0.04180392,  0.20834295,  0.05230585, -0.01441469,  0.08332603,
       -0.00502916, -0.07292716, -0.06649075, -0.12857822, -0.00481425,
        0.07093009, -0.07851573,  0.0155651 ,  0.12625268, -0.15305419,
        0.24509557,  0.02032769, -0.06740412, -0.06718043, -0.03178652,
       -0.09728859,  0.12896375,  0.1560752 , -0.31920522,  0.22090906,
        0.16365205,  0.02951704,  0.17214292,  0.12328891, -0.00719389,
        0.0128033 , -0.06052898, -0.17239068, -0.11195953,  0.0419649 ,
       -0.01805997,  0.06162161,  0.04654519])
A = np.array([x0, x1])
B = A
print(A)
norm_numpy = np.linalg.norm(x0-x1)
print("Norma numpy = " +str(norm_numpy))
norm_torch = pairwise_euclidean_distance(torch.from_numpy(A), torch.from_numpy(B))
print("Norma torch meva = " +str(norm_torch))