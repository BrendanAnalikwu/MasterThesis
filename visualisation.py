import glob
import os
import re
from collections import defaultdict
from typing import List, Tuple, Union, Dict

import matplotlib.pyplot as plt
import torch.nn
import numpy as np
from matplotlib.colors import SymLogNorm
from torchvision.utils import make_grid

from dataset import transform_data, BenchData, SeaIceDataset


def plot_comparison(model: torch.nn.Module, dataset: BenchData, i: int = 0, channel: int = 0, patch_size: int = 3,
                    overlap: int = 1):
    model.eval()
    output = model(*dataset[i:(i+1)][:-1]).detach()

    fig, ax = plt.subplots(1, 2)
    vmin = min(output[channel].min(), dataset.label[i, channel].min())
    vmax = max(output[channel].max(), dataset.label[i, channel].max())

    im = ax[0].imshow(output.cpu()[0][channel], norm=SymLogNorm(linthresh=1e-2, linscale=1, vmin=vmin, vmax=vmax, base=10))
    ax[1].imshow(dataset.label[i, channel].cpu(), norm=SymLogNorm(linthresh=1e-2, linscale=1, vmin=vmin, vmax=vmax, base=10))
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


def plot_experiments_loss(experiments: List[str], loss_type: str = 'classic', last: bool = False):
    res = defaultdict(lambda: [])
    for exp in experiments:
        for fn in glob.glob(f'experiments/{exp}/' + r'*.li'):
            losses = torch.load(fn)
            res[exp].append(losses[loss_type][-1] if last else min(losses[loss_type]))

    mean = [np.mean(li) for li in res.values()]
    # TODO: compute errorbar

    plt.bar(experiments, mean)
    for i, exp in enumerate(experiments):
        plt.scatter([i] * len(res[exp]), res[exp], c='k')


def load_model(exp: str, i: int):
    if exp[-3:] == '.pt':
        return torch.load('experiments/' + exp)
    else:
        fn = ['experiments/' + exp + '/' + f for f in os.listdir('experiments/' + exp) if f[-3:] == '.pt']
        return torch.load(fn[i])


def plot_model_output(model: torch.nn.Module, dataset: SeaIceDataset, i: int = 0, dim: int = 0):
    plt.figure()
    output = model(*dataset[i:(i+1)][:-1]).detach().cpu()[0, dim]
    plt.imshow(output, norm=SymLogNorm(linthresh=1e-2, linscale=1, base=10))
    plt.show()


if __name__ == "__main__":
    plot_results('arch', [3, 5], [1, 2, 3, 4, 5, 6, 7])
    plot_results('arch', [5], [2, 3, 4, 5, 6, 7], c=1)
    plot_results('arch', [5], [2, 3, 4, 5, 6, 7], c=1, h=32)
    plot_results('arch', [5], [1, 2, 3, 4, 5, 6, 7], c=0, h=8)
