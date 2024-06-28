import sys
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, _check_length_scale, Kernel


def get_scores():
    pass


def save_params():
    pass


class kernel(RBF):
    def __init__(self, length_scale=1.0):
        super().__init__(length_scale)

    ranges = np.array([(-3, -.5), (-4, -1.3), (3, 7), (-2, 3), (-3, -.5), (-6, -2), (-4, 1), (-6, -1)])
    scales = np.array([b - a for a, b in ranges])

    @staticmethod
    def distance(u: np.array, v: np.array):
        res = (((u - v)[2:-2] / kernel.scales[:-2]) ** 2).sum()
        res += ((u[:2] != v[:2]) * 4.).sum()
        if u[1] == v[1]:
            if u[1] == 3:  # MSE+SRE
                res += ((u[-2] - v[-2]) / kernel.scales[-2]) ** 2
            elif u[1] == 4:  # MSE+MRE
                res += (((u[-2:] - v[-2:]) / kernel.scales[-2:]) ** 2).sum() / np.sqrt(2)
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


def acquisition(model, ij):
    def func(x, *args):
        mean, std = model.predict(np.atleast_2d(np.hstack((ij, x))), return_std=True)
        return mean - 1.96 * std

    return func


models = ['SurrogateNet', 'UNet']
loss = ['MSE', 'SRE', 'MSE+SRE', 'MSE+MRE', 'MAE']


def find_min(model):
    res = []
    for i in range(len(models)):
        for j in range(len(loss)):
            x0 = np.empty(10)
            x0[:2] = i, j
            x0[2:] = np.random.uniform(0, 1, 8) * kernel.scales + kernel.ranges[:, 0]
            r = minimize(acquisition(model, x0[:2]), x0[2:], 'Nelder-Mead', bounds=list(kernel.ranges))
            r.x = np.hstack((x0[:2], r.x))
            res.append(r)
    return res[np.argmin([r.fun for r in res])].x


if __name__ == "__main__":
    # Load scores from register
    data_scores = np.atleast_2d(np.loadtxt('register.txt'))  # id, score
    data_scores = data_scores[data_scores[:, 0].argsort()]
    data_params = np.atleast_2d(np.loadtxt('running.txt'))  # id, x
    data_params = data_params[data_params[:, 0].argsort()]

    if len(data_scores) < 4:
        x_new = np.hstack((np.array([np.random.randint(2), np.random.randint(5)]),
                           np.random.uniform(0, 1, 8) * kernel.scales + kernel.ranges[:, 0]))
    else:
        running_ids = np.setdiff1d(data_params[:, 0], data_scores[:, 0])
        X = data_params[np.isin(data_params[:, 0], data_scores[:, 0]), -10:]  # Get parameters
        scores = np.log10(data_scores[:, 1])  # Get scores and convert to log10 base

        # Train GP model
        GP = GaussianProcessRegressor(kernel=kernel(2.), alpha=1e-4 ** 2, optimizer=None, normalize_y=True)
        GP.fit(X, np.log10(scores))

        # Fill with fakes
        if len(running_ids) > 0:
            X_running = data_params[np.isin(data_params[:, 0], running_ids), -10:]
            m, s = GP.predict(X_running, return_std=True)
            fake_scores = (m + .5 * s)[:, None]
            GP.fit(np.vstack((X, X_running)), np.vstack((scores, fake_scores)))

        # Find next parameter set
        x_new = find_min(GP)

    # Save parameters to register
    if len(sys.argv) > 1:
        id_new = sys.argv[1]
    elif data_params[:, 0].size == 0:
        id_new = 0
    else:
        id_new = max(data_params[:, 0]) + 1

    with open('running.txt', 'ab') as f:
        np.savetxt(f, np.hstack((float(id_new), x_new))[None], fmt="%d %d %d %f %f %f %f %f %f %f %f")
    sys.stdout.write(' '.join(str(int(x)) for x in x_new[:2]) + ' ' + ' '.join(str(x) for x in x_new[2:]))
    sys.exit(0)
