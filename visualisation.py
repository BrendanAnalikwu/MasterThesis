import matplotlib.pyplot as plt
import torch.nn
from matplotlib.colors import SymLogNorm
from torchvision.utils import make_grid

from dataset import transform_data, BenchData


def plot_comparison(model: torch.nn.Module, dataset: BenchData, i: int, channel: int = 0, chunk_size: int = 17,
                    overlap: int = 2):
    d = transform_data(*dataset.retrieve(i), chunk_size, overlap)
    model.eval()
    num_chunks = int(255 / (chunk_size - overlap))
    c, m, s = model(*d[:-1])[:num_chunks ** 2]
    res = make_grid(c * s + m, num_chunks, padding=0).detach()
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
