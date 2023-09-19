import os
from math import ceil
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import trange
import matplotlib.pyplot as plt

from dataset import BenchData
from surrogate_net import PatchNet
from visualisation import plot_comparison, plot_losses
from torchvision.utils import make_grid
from dataset import transform_data

dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Constants
L = 1e6
T = 1e3
G = 1.
C_o = 1026 * 5.5e-3 * L / (900 * G)
C_a = 1.3 * 1.2e-3 * L / (900 * G)
C_r = 27.5e3 * T ** 2 / (2 * 900 * L ** 2)
e_2 = .25
C = 20
f_c = 1.46e-4
dx = 512e3 / 256 / L  # 4km
dT = 1e3 / T  # 1000s

patch_size, overlap = 3, 5
model = PatchNet(overlap, patch_size, (1, 2, 3)).to(dev)

if os.path.isfile(f'full_dataset_{patch_size}-{overlap}.data'):
    dataset = torch.load(f'full_dataset_{patch_size}-{overlap}.data')
else:
    dataset = BenchData("C:\\Users\\Brend\\Thesis\\GAS\\seaice\\benchmark\\Results8\\", list(range(1, 97)), patch_size,
                        overlap, dev=dev)
    torch.save(dataset, f'full_dataset_{patch_size}-{overlap}.data')
dataloader = DataLoader(dataset, batch_size=ceil(len(dataset) / 10), shuffle=True)

criterion = torch.nn.MSELoss().to(dev)
structure_criterion = torch.nn.MSELoss().to(dev)
inst_norm = torch.nn.InstanceNorm2d(2).to(dev)
optim = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(.95, .995))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[12000], gamma=.1)
losses = []
mean_losses = []
std_losses = []
contrast_losses = []
classic_losses = []

pbar = range(128)
for i in pbar:
    for (data, H, A, v_a, v_o, border_chunk, label) in dataloader:
        # Forward pass and compute output
        output = model(data, H, A, v_a, v_o, border_chunk)
        contrast = inst_norm(output)
        m = output.mean(dim=(2, 3), keepdim=True)
        s = output.std(dim=(2, 3), keepdim=True, unbiased=False)

        # Compute losses
        contrast_loss = structure_criterion(contrast, inst_norm(label))
        classic_loss = criterion(output, label)
        loss = 1e4 * classic_loss + contrast_loss
        with torch.no_grad():
            mean_loss = (m - label.mean(dim=(2, 3), keepdim=True)).square().mean()
            std_loss = (s - label.std(dim=(2, 3), keepdim=True, unbiased=False)).square().mean()

        # Store losses
        losses.append(loss.item())
        mean_losses.append(mean_loss.item())
        std_losses.append(std_loss.item())
        contrast_losses.append(contrast_loss.item())
        classic_losses.append(classic_loss.item())

        # Gradient computation and optimiser step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optim.step()
        torch.cuda.empty_cache()
        scheduler.step()
    # pbar.set_postfix(loss=losses[-1], contrast_loss=contrast_losses[-1])  # , PINN=PINN_losses[-1])

torch.save(model, f'model_{patch_size}-{overlap}.pt')
torch.save({'loss': losses, 'mean': mean_losses, 'std': std_losses, 'contrast': contrast_losses, 'classic': classic_losses}, 'losses.li')

# Plot results
# plot_comparison(model, dataset, 0, 0, patch_size, overlap)
# plot_losses(losses, mean_losses, std_losses, contrast_losses, classic_losses,
#             names=['loss', 'mean', 'std', 'contrast', 'classic'])
print('done')
