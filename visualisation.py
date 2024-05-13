import glob
import os
import re
from collections import defaultdict
from math import erf
from typing import List, Tuple, Union, Dict

import matplotlib.pyplot as plt
import torch.nn
import numpy as np
from matplotlib.colors import SymLogNorm, Normalize
from torchvision.utils import make_grid

from dataset import transform_data, BenchData, SeaIceDataset, SeaIceTransform, FourierData
from generate_data import read_vtk2


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


def load_model(exp: str, i: int, dev='cpu'):
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


def plot_output(exp: str, i: int = 0, img: int = 0, linthresh=1e-2):
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


if __name__ == "__main__":
    dev = torch.device('cpu')
    dataset = FourierData("C:/Users/Brend/PycharmProjects/MasterThesis/data/test", dev=dev, phys_i=10,
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