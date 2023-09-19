from typing import List

import matplotlib.pyplot as plt
import torch.nn
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
