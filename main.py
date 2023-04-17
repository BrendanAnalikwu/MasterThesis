import glob
from typing import Tuple
import torch
import torch.utils.data
import numpy as np
from tqdm import trange


class SinActivation(torch.nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return torch.sin(x)


class HeatNet(torch.nn.Module):
    def __init__(self, Nx, Ny, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = torch.nn.Sequential(
            # Input: Nx(Nx+Ny)
            torch.nn.Linear(Nx + Ny, 32),
            torch.nn.LeakyReLU(.1),
            # N x 32
            torch.nn.Linear(32, 128),
            torch.nn.LeakyReLU(.1),
            # N x 64
            torch.nn.Unflatten(-1, (8, 4, 4)),
            # N x 8x4x4
            torch.nn.ConvTranspose2d(8, 8, 3, 1, 1),
            torch.nn.ConvTranspose2d(8, 8, 3, 2, 0),
            torch.nn.ReLU(),
            # N x 8x9x9
            torch.nn.ConvTranspose2d(8, 4, 3, 1, 1),
            torch.nn.ConvTranspose2d(4, 4, 3, 2, 1),
            torch.nn.ReLU(),
            # N x 4x17x17
            torch.nn.ConvTranspose2d(4, 4, 3, 1, 1),
            torch.nn.ConvTranspose2d(4, 4, 3, 2, 1),
            torch.nn.ReLU(),
            # N x 4x33x33
            torch.nn.ConvTranspose2d(4, 4, 3, 1, 1),
            torch.nn.ConvTranspose2d(4, 1, 3, 1, 1),
            # N x 1x33x33
            torch.nn.Tanh()
        )

    def forward(self, x: torch.Tensor):
        return self.model(x).squeeze()


def get_dataset(Nx: int, Ny: int, name='latest') -> torch.utils.data.Dataset:
    if name == 'latest':
        name = max([int(s.strip('dataset_8_8_33x33_.gz')) for s in glob.glob('dataset_8_8_33x33_*.gz')])
    raw_data = np.loadtxt(f'dataset_8_8_33x33_{name}.gz', dtype=np.float32)
    return torch.utils.data.TensorDataset(torch.tensor(raw_data[:, :Nx + Ny]),
                                          torch.tensor(raw_data[:, Nx + Ny:]).reshape(-1, 33, 33))


Nx = 8
Ny = 8
dataset = get_dataset(Nx, Ny)
loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)

model = HeatNet(Nx, Ny)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_func = torch.nn.MSELoss()

epochs = 20
losses = []
for epoch in trange(epochs):
    for input, label in loader:
        res = model(input)
        loss = loss_func(res, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

test_data, test_labels = get_dataset(Nx, Ny, '230415182814')[:]
print(f"Test loss: {loss_func(test_labels, model(test_data)).item()}")

print('done')
