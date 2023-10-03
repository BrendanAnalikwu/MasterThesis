import glob
from typing import List

import matplotlib.pyplot as plt
import torch.nn
import numpy as np
from matplotlib.colors import SymLogNorm
from torchvision.utils import make_grid

from dataset import transform_data, BenchData


def plot_comparison(model: torch.nn.Module, dataset: BenchData, i: int, channel: int = 0, patch_size: int = 3,
                    overlap: int = 1):
    d = transform_data(*dataset.retrieve(i), patch_size, overlap)
    model.eval()
    num_chunks = int(255 / patch_size)
    output = model(*d[:-1])[:num_chunks ** 2]
    res = make_grid(output, num_chunks, padding=0).detach()
    model.train()

    fig, ax = plt.subplots(1, 2)
    vmin = min(res[channel].min(), dataset.label[i, channel].min())
    vmax = max(res[channel].max(), dataset.label[i, channel].max())

    im = ax[0].imshow(res.cpu()[channel], norm=SymLogNorm(linthresh=1e-2, linscale=1, vmin=vmin, vmax=vmax, base=10))
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


def plot_results():
    res = {}
    for j in (3, 5, 15):
        res[j] = []
        for i in (1, 2, 3, 5, 9, 15):
            li = []
            for fn in glob.glob(f'patch_size_experiment/losses/losses_{j}-{i}*.li'):
                li.append(torch.load(fn)['classic'][-1])
            res[j].append((sum(li) / len(li), li))
            # plt.scatter([i] * len(li), li)
        plt.errorbar([1, 2, 3, 5, 9, 15], [res[j][l][0] for l in range(len(res[j]))], yerr=[np.std(res[j][l][1]) for l in range(len(res[j]))], capsize=5)

    plt.legend([3, 5, 15])
    plt.xticks([1, 2, 3, 5, 9, 15], [1, 2, 3, 5, 9, 15])
    plt.show()


if __name__ == "__main__":
    print('plotting results')
    plot_results()
    print('done')
