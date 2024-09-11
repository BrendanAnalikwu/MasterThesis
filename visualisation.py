import glob
import os
import re
from collections import defaultdict
from math import erf
from typing import List, Tuple, Union, Dict

import matplotlib.pyplot as plt
import torch.nn
import numpy as np
from matplotlib.colors import SymLogNorm, Normalize, LinearSegmentedColormap
from torchvision.utils import make_grid

from bayesian_optimization import param_names
from dataset import transform_data, BenchData, SeaIceDataset, SeaIceTransform, FourierData
from generate_data import read_vtk2
from loss import advect

cmap_bbw = LinearSegmentedColormap.from_list('bbw', ['black', 'blue', 'white'])


def plot_comparison(model: torch.nn.Module, dataset: BenchData, i: int = 0, channel: int = 0, normed: bool = False):
    model.eval()
    output = model(*dataset.retrieve(slice(i, i + 1))[:-1]).detach()

    fig, ax = plt.subplots(1, 2)
    vmin = min(output[0, channel].min(), dataset.label[i, channel].min())
    vmax = max(output[0, channel].max(), dataset.label[i, channel].max())
    if normed:
        norm = SymLogNorm(linthresh=1e-2, linscale=1, vmin=vmin, vmax=vmax, base=10)
    else:
        norm = Normalize(vmin, vmax)

    im = ax[0].imshow(output.cpu()[0, channel], norm=norm)
    ax[1].imshow(dataset.label[i, channel].cpu(), norm=norm)
    fig.tight_layout()
    fig.subplots_adjust(right=.85)
    cbar_ax = fig.add_axes([.9, .15, .03, .7])
    fig.colorbar(im, cax=cbar_ax)


def plot_losses(*args: List, **kwargs):
    plt.figure()
    for arg in args:
        plt.semilogy(arg)
    if 'names' in kwargs.keys() and len(kwargs['names']) == len(args):
        plt.legend(kwargs['names'])


def plot_all_losses(losses: dict[list]):
    plot_losses(*losses.values(), names=losses.keys())


def plot_results(experiment: str, p: Union[Tuple, List] = (3, 5, 15), o: Union[Tuple, List] = (1, 2, 3, 5, 7, 9),
                 c: int = 0, h: int = 16):
    dir = f'experiments/{experiment}/'
    plt.figure()
    res = {}
    for j in p:
        res[j] = {}
        for i in o:
            li = []
            files = filter(re.compile(f'losses_{j}-{i}-\(8,{h},\d*\)' + (r'(-0)*' if c == 0 else f'-{c}*') + r'-256_.*li').match, os.listdir(dir))
            for fn in files:
                li.append(min(torch.load(dir + fn)['classic']))
            res[j][i] = (sum(li) / len(li) if len(li) != 0 else 0, li)
            plt.scatter([i+j*.05-.15] * len(li), li, c='k')
        plt.errorbar([q + j*.05-.15 for q in o], [res[j][l][0] for l in o], yerr=[np.std(res[j][l][1])/np.sqrt(len(res[j][l][1])) for l in o], capsize=5)

    plt.legend(p)
    plt.xticks(o, o)
    return res


def plot_experiments_loss(experiments: List[str], loss_type: str = 'classic', last: bool = False, names = None):
    res = defaultdict(lambda: [])
    for exp in experiments:
        for fn in glob.glob(f'experiments/{exp}/' + r'*.li'):
            losses = torch.load(fn)
            res[exp].append(losses[loss_type][-1] if last else min(losses[loss_type]))

    mean = [np.mean(li) for li in res.values()]
    # TODO: compute errorbar

    plt.bar(experiments if names is None else names, mean)
    for i, exp in enumerate(experiments):
        plt.scatter([i] * len(res[exp]), res[exp], c='k')


def load_model(exp: str, i: int = None, dev='cpu'):
    if exp[-3:] == '.pt':
        return torch.load('experiments/' + exp).to(dev).eval()
    else:
        fn = ['experiments/' + exp + '/' + f for f in os.listdir('experiments/' + exp) if f[-3:] == '.pt']
        return torch.load(fn[i]).to(dev).eval()


def load_losses(exp: str, i: int = None):
    if exp[-3:] == '.li':
        return torch.load('experiments/' + exp)
    elif i is not None:
        fn = ['experiments/' + exp + '/' + f for f in os.listdir('experiments/' + exp) if f[-3:] == '.li']
        return torch.load(fn[i])
    else:
        raise ValueError("Either give a file name ending in 'li', or a directory and number")


def plot_model_output(model: torch.nn.Module, dataset: SeaIceDataset, i: int = 0, dim: int = 0):
    plt.figure()
    output = model(*dataset[i:(i+1)][:-1]).detach().cpu()[0, dim]
    plt.imshow(output)#, norm=SymLogNorm(linthresh=1e-2, linscale=1, base=10))
    plt.show()


def p_value(exp1: str, exp2: str, loss_type: str = 'MCE'):
    fn = ['experiments/' + exp1 + '/' + f for f in os.listdir('experiments/' + exp1) if f[-3:] == '.li']
    l1 = [np.mean(torch.load(file)[loss_type][-1]) for file in fn]
    fn = ['experiments/' + exp2 + '/' + f for f in os.listdir('experiments/' + exp2) if f[-3:] == '.li']
    l2 = [np.mean(torch.load(file)[loss_type][-1]) for file in fn]
    l0 = l1 + l2
    z = (np.abs(np.mean(l1) - np.mean(l2))) / np.sqrt(1/len(l1) + 1/len(l2)) / np.std(l0)
    p = (1 - erf(z)) / 2
    return p


def plot_losses_avg(exp: str, i: int = 0, loss = 'MSE'):
    losses = load_losses(exp, i)
    tlosses = load_losses(f'{exp}/test', i)
    plt.figure()
    plt.semilogy([np.mean(losses[loss][i:i + 413]) for i in range(len(losses[loss]) - 413)])
    plt.semilogy(np.linspace(0, len(losses[loss])-413, len(tlosses[loss])), tlosses[loss])
    plt.legend(['avg train', 'test'])


def get_test_indices(names: str, exp: str, i: int = 0):
    with open(glob.glob(f'experiments/{exp}/' + r'/test_data_*.txt')[i], 'r', encoding='latin-1') as file:
        test_names = file.read().split()
    return [np.where(n == np.array(names))[0][0]for n in np.intersect1d(names, test_names)]


def plot_output(exp: str, i: int = 0, img: int = 0, dataset=None, linthresh=1e-2):
    model = load_model(exp, i)
    output = model(*dataset[::20][:-1]).detach()[img, 0]
    label = dataset.label[::20][img, 0]
    vmin = min(output.min(), label.min())
    vmax = max(output.max(), label.max())
    norm = SymLogNorm(linthresh, vmin=vmin, vmax=vmax)
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(label, norm=norm)
    ax[1].imshow(output, norm=norm)


is_close_fraction = lambda p: 1 - torch.isclose(p.abs().cpu().detach(), torch.tensor(0.), atol=1e-6).sum() / p.numel()
absolute_size = lambda p: p.abs().sum().cpu().detach()
mean_size = lambda p: p.abs().mean().cpu().detach()


def plot_parameters(exp: str, i: int = 0, mode=mean_size, fig: int = None, c='None'):
    plt.figure(fig)
    plt.semilogy([mode(p) if abs else is_close_fraction(p) for p in load_model(exp, i).parameters() if p.dim() > 3], c=c)


def require_gradient(dataset: SeaIceDataset):
    dataset.data.requires_grad = True
    dataset.H.requires_grad = True
    dataset.A.requires_grad = True
    dataset.v_a.requires_grad = True


def compute_sensitivity(model: torch.nn.Module, im: int):
    output = model(*dataset[slice(im, im + 1)][:-2])
    gg = sum([torch.autograd.grad(output[0, 0, i, j], dataset.data, retain_graph=True)[0][im].abs().sum(dim=0) for i in range(257) for j in range(257)])


def propagate(model, dataset: FourierData, v0, H0, A0, v_a, v_o):
    assert len(v_a) == len(v_o)
    n_steps = len(v_a)
    model.eval()
    v = torch.empty((n_steps + 1, 2, *v0.shape[-2:]))
    v[0] = dataset.data_scaling.inverse(v0)
    H = torch.empty((n_steps + 1, 1, *H0.shape[-2:]))
    H[0] = dataset.H_scaling.inverse(H0)
    A = torch.empty((n_steps + 1, 1, *H0.shape[-2:]))
    A[0] = dataset.A_scaling.inverse(A0)

    for i in range(n_steps):
        out = model(dataset.data_scaling(v[i, None]),
                    dataset.H_scaling(H[i, None]),
                    dataset.A_scaling(A[i, None]),
                    dataset.v_a_scaling(v_a[i, None]),
                    v_o[i, None])
        v[i + 1] = v[i] + dataset.label_scaling.inverse(out.detach()[0])
        H[i + 1] = advect(v[i + 1, None], H[i, None], 2., .5 / 256, lb=0.)
        A[i + 1] = advect(v[i + 1, None], A[i, None], 2., .5 / 256, lb=0., ub=1.)

    return v, H, A


def im_to_disc(x):
    with open('guess.vec', 'w') as f:
        f.write("66049 2")
        # Get order
        # Sort order
        # Reorder our field
        # Write items

        f.write('BackUpEnd')


def bo_scatter():
    data_scores = np.atleast_2d(np.loadtxt('register.txt'))  # id, score
    data_scores = data_scores[data_scores[:, 0].argsort()]
    data_params = np.atleast_2d(np.loadtxt('running.txt'))  # id, x
    data_params = data_params[data_params[:, 0].argsort()]
    params = data_params[np.isin(data_params[:, 0], data_scores[:, 0]), -10:]
    scores = data_scores[:, 1]

    plt.figure()
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.scatter(params[:, i + 2], scores)
        plt.title(param_names[i + 2])


def save_concentration(fn: str, im: torch.Tensor, bench=False):
    fig = plt.figure(frameon=False, figsize=(4, 4))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(im.T.flip([0, 1]) if bench else im.T, cmap=cmap_bbw, vmax=1, vmin=.75, origin='lower')
    fig.savefig('plots/' + fn + '.pdf')


def save_velocity(fn: str, im: torch.Tensor, vm=None, bench=False):
    fig = plt.figure(frameon=False, figsize=(4, 4))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    if not vm:
        vm = im.abs().max()
    ax.imshow(SeaIceTransform.transform_velocity(im.T, 2, 0) if bench else im.T, cmap='seismic', vmax=vm, vmin=-vm, origin='lower')
    fig.savefig('plots/' + fn + f'[{vm:.2e}].pdf')


def save_shear(fn: str, im: torch.Tensor, vmin: float = None, vmax: float = None, bench=False):
    fig = plt.figure(frameon=False, figsize=(4, 4))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    if not vmin:
        vmin = im.min()
    if not vmax:
        vmax = im.max()
    ax.imshow(im.T.flip([0, 1]) if bench else im.T, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
    fig.savefig('plots/' + fn + f'[{vmin:.2e}, {vmax:.2e}].pdf')


if __name__ == "__main__":
    dev = torch.device('cpu')
    dataset = FourierData("C:/Users/Brend/PycharmProjects/MasterThesis/data/test", dev=dev, phys_i=10,
                          scaling={'data': (0., torch.tensor([[[[0.0004184620047453791]], [[0.0003502570034470409]]]])),
                                   'label': (0., torch.tensor([[[[0.0001648720062803477]], [[0.00017528010357636958]]]])),
                                   'v_a': (0., torch.tensor([[[[0.018403656780719757]], [[0.018403703346848488]]]])),
                                   'H': (0., torch.tensor([[[[1.0179400444030762]]]])),
                                   'A': (0., torch.tensor([[[[1.]]]]))})
    bench = FourierData("C:/Users/Brend/PycharmProjects/MasterThesis/data/bench", dev=dev, phys_i=10,
                        scaling={'data': (0., torch.tensor([[[[0.0004184620047453791]], [[0.0003502570034470409]]]])),
                                 'label': (0., torch.tensor([[[[0.0001648720062803477]], [[0.00017528010357636958]]]])),
                                 'v_a': (0., torch.tensor([[[[0.018403656780719757]], [[0.018403703346848488]]]])),
                                 'H': (0., torch.tensor([[[[1.0179400444030762]]]])),
                                 'A': (0., torch.tensor([[[[1.]]]]))})
    # benchmark = BenchData("C:/Users/Brend/Thesis/GAS/seaice/benchmark/Results8/", list(range(1, 97)), dev)

    print('done')

# f, ax = plt.subplots(1,2)
# ax[0].imshow(dataset.label_scaling.inverse(output).detach()[80,0] * 1e-4, norm=norm)
# im = ax[1].imshow(dataset.label_scaling.inverse(dataset.label).detach()[80,0] * 1e-4, norm=norm)
# plt.tight_layout()
# f.subplots_adjust(right=.85)
# cbar = f.add_axes([.9,.15,.03,.7])
# cb = f.colorbar(im, cax=cbar, ticks = [-5e-6, -1e-6, -5e-7, 0, 5e-7, 1e-6, 5e-6], format=ticker.LogFormatterMathtext(labelOnlyBase=False))
# cb.set_ticklabels([-5e-6, -1e-6, -5e-7, 0, 5e-7, 1e-6, 5e-6])
# cb.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

# model = load_model('time', 3)
#
#
# output = model(*dataset[219:220][:-2])
#
# gg = sum([torch.autograd.grad(output[0,0,i,j], dataset.data, retain_graph=True)[0][219].abs().sum(dim=0) for i in range(257) for j in range(257)])
