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


if __name__ == "__main__":
    dev = torch.device('cpu')
    dataset = FourierData("C:/Users/Brend/PycharmProjects/MasterThesis/data/data/", dev=dev)
    benchmark = BenchData("C:/Users/Brend/Thesis/GAS/seaice/benchmark/Results8/", list(range(1, 97)), dev)

    print('done')
