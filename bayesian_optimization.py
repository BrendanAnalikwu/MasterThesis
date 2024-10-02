import sys
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, _check_length_scale, Kernel

import matplotlib.pyplot as plt


def get_scores():
    pass


def save_params():
    pass


class kernel(RBF):
    def __init__(self, length_scale=1.0):
        super().__init__(length_scale)

    @staticmethod
    def distance(u: np.array, v: np.array):
        res = (((u - v)[2:-2] / scales[:-2]) ** 2).sum()
        res += ((u[:2] != v[:2]) * 4.).sum()
        if u[1] == v[1]:
            if u[1] == 3:  # MSE+SRE
                res += ((u[-2] - v[-2]) / scales[-2]) ** 2
            elif u[1] == 4:  # MSE+MRE
                res += (((u[-2:] - v[-2:]) / scales[-2:]) ** 2).sum() / np.sqrt(2)
            # Otherwise no extra distance
        return res

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            dists = pdist(X, metric=kernel.distance)
            K = np.exp(-.5 * dists * length_scale ** 2)
            K = squareform(K)
            np.fill_diagonal(K, 1.)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            dists = cdist(X, Y, metric=kernel.distance)
            K = np.exp(-.5 * dists * length_scale ** 2)

        if eval_gradient:
            K_gradient = (K * squareform(dists))[:, :, np.newaxis]
            return K, K_gradient
        else:
            return K


def acquisition(model):
    def func(x, *args):
        mean, std = model.predict(np.atleast_2d(x), return_std=True)
        return mean - 1.96 * std

    return func


models = ['UNet', 'SurrogateNet']
loss = ['MAE', 'MSE', 'SRE', 'MSE+SRE', 'MSE+MRE']
param_names = ['alpha', 'noise_lvl', 'lr', 'weight']
ranges = np.array([(-3, 2), (-3, -.5), (-4, -1), (-1, 3)])
scales = np.array([b - a for a, b in ranges])


def find_min(model):
    res = []
    for i in range(4**len(param_names)):
        x0 = np.random.uniform(0, 1, len(param_names)) * scales + ranges[:, 0]
        r = minimize(acquisition(model), x0, 'Nelder-Mead', bounds=list(ranges))
        res.append(r)
    return res[np.argmin([r.fun for r in res])].x


if __name__ == "__main__":
    # Load scores from register
    data_scores = np.loadtxt('register.txt').reshape(-1, 2)  # id, score
    data_scores = data_scores[data_scores[:, 0].argsort()]
    data_params = np.loadtxt('running.txt').reshape(-1, len(param_names) + 1)  # id, x

    if len(data_scores) < 10:
        x_new = np.random.uniform(0, 1, len(param_names)) * scales + ranges[:, 0]
    else:
        data_params = data_params[data_params[:, 0].argsort()]
        running_ids = np.setdiff1d(data_params[:, 0], data_scores[:, 0])
        X = data_params[np.isin(data_params[:, 0], data_scores[:, 0]), -len(param_names):]  # Get parameters
        scores = np.log10(data_scores[:, 1])  # Get scores and convert to log10 base

        # Train GP model
        GP = GaussianProcessRegressor(kernel=RBF(1., (1e-2, 1e2)), optimizer=None, alpha=1e-2, normalize_y=True)
        GP.fit(X, scores)

        # Fill with fakes
        if len(running_ids) > 10:
            X_running = data_params[np.isin(data_params[:, 0], running_ids), -len(param_names):]
            fake_scores = GP.predict(X_running)
            GP.fit(np.vstack((X, X_running)), np.hstack((scores, fake_scores)))

        # Find next parameter set
        x_new = find_min(GP)

    # Save parameters to register
    if len(sys.argv) > 1:
        id_new = sys.argv[1]
    elif data_params[:, 0].size == 0:
        id_new = 0
    else:
        id_new = max(data_params[:, 0]) + 1

    with open('running.txt', 'a') as f:
        f.write(("%s" + " %f" * len(x_new) + "\n") % (id_new, *tuple(x_new)))
    sys.stdout.write(' '.join(str(x) for x in x_new))
    sys.exit(0)

# for j in range(len(param_names)):
#     i = np.argmin(scores)
#     x = np.empty((1000, len(param_names)))
#     x[:] = X[i, :]
#     x[:, j] = np.linspace(ranges[j][0], ranges[j][1], 1000)
#     m, s = GP.predict(x, return_std=True)
#     plt.figure()
#     plt.fill_between(x[:, j], m - 1.96 * s, m + 1.96 * s, alpha=.2)
#     plt.scatter(X[:, j], scores, alpha=np.maximum(RBF(1.)(X[i], X), .1))
#     plt.plot(x[:, j], m)
# plt.show()