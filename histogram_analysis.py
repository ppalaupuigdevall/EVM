import numpy as np
import matplotlib.pyplot as plt

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

persona = np.load('Alberto_Garzon.npy')
persona = np.log(1.0+persona)
for i in range(0, np.shape(persona)[0]):
	d_sorted = 0.5 * np.sort(persona[i, :])[:300]
	k_i, lambda_i = fit_(d_sorted, iters = 100, eps = 1e-6)
	print("K_i = " + str(k_i) + ", lambda_i = " + str(lambda_i))
	plt.subplot(211)
	values, bins_hist, patches = plt.hist(d_sorted, bins = len(d_sorted), density = True)
	plt.subplot(212)
	x_axis = np.linspace(np.min(bins_hist), np.max(bins_hist), num = len(d_sorted))

	plt.plot(x_axis, np.exp(-((x_axis/lambda_i)**k_i)))
	plt.show()

print(np.shape(persona))

# Pintar intra i inter, un per intra (mateixa classe) i l'altre per inter. 
# Provar a estimar els weibull amb les intra i FitHigh?