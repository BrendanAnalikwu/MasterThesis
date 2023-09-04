import matplotlib.pyplot as plt
import torch.nn
from matplotlib.colors import SymLogNorm
from torchvision.utils import make_grid

from dataset import transform_data, BenchData


def plot_comparison(model: torch.nn.Module, dataset: BenchData, i: int, channel: int = 0):
    d = transform_data(*dataset.retrieve(i), 5, 2)
    model.eval()
    res = make_grid(model(*d[:-1])[:85 ** 2].detach(), 85, padding=0)
    model.train()

    fig, ax = plt.subplots(1, 2)
    vmin = min(res[0].min(), dataset.label[i, channel].min())
    vmax = max(res[0].max(), dataset.label[i, channel].max())

    im = ax[0].imshow(res.cpu()[channel], norm=SymLogNorm(linthresh=1e-3, linscale=1, vmin=vmin, vmax=vmax, base=10))
    ax[1].imshow(dataset.label[i, channel].cpu(), norm=SymLogNorm(linthresh=1e-3, linscale=1, vmin=vmin, vmax=vmax, base=10))
    fig.tight_layout()
    fig.subplots_adjust(right=.85)
    cbar_ax = fig.add_axes([.9, .15, .03, .7])
    fig.colorbar(im, cax=cbar_ax)
