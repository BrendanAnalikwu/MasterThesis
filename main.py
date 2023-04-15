import glob
from typing import Tuple

import torch
import numpy as np


class HeatNet(torch.nn.Module):
    def __init__(self, Nx, Ny, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = torch.nn.Sequential(
            # Input: Nx(Nx+Ny)
            torch.nn.Linear(Nx + Ny, 32),
            torch.nn.LeakyReLU(.1),
            # N x 32
            torch.nn.Linear(32, 64),
            torch.nn.LeakyReLU(.1),
            # N x 64
            torch.nn.Unflatten(-1, (4, 4, 4)),
            # N x 4x4x4
            torch.nn.ConvTranspose2d(4, 4, 3, 1, 1),
            torch.nn.ConvTranspose2d(4, 4, 3, 2, 0),
            torch.nn.ReLU(),
            # N x 4x9x9
            torch.nn.ConvTranspose2d(4, 4, 3, 1, 1),
            torch.nn.ConvTranspose2d(4, 4, 3, 2, 1),
            torch.nn.ReLU(),
            # N x 4x17x17
            torch.nn.ConvTranspose2d(4, 4, 3, 1, 1),
            torch.nn.ConvTranspose2d(4, 4, 3, 2, 0),
            torch.nn.ReLU(),
            # N x 4x33x33
            torch.nn.ConvTranspose2d(4, 4, 3, 1, 1),
            torch.nn.ConvTranspose2d(4, 1, 3, 1, 1),
            # N x 1x33x33
            torch.nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)


def get_dataset(Nx: int, Ny: int, name='latest') -> Tuple[torch.Tensor, torch.Tensor]:
    if name == 'latest':
        name = max([int(s.strip('dataset_8_8_33x33_.gz')) for s in glob.glob('dataset_8_8_33x33_*.gz')])
    raw_data = np.loadtxt(f'dataset_8_8_33x33_{name}.gz')
    return torch.tensor(raw_data[:, :Nx + Ny]), torch.tensor(raw_data[:, Nx + Ny:]).reshape(-1, 33, 33)


data, labels = get_dataset(8, 8)
print(f'Number of samples in dataset is {len(data)}')

print('done')
