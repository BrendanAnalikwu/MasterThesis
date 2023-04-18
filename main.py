import glob
from typing import Tuple
import torch
import torch.utils.data
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt


class SinActivation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return torch.sin(x)


class HeatNet(torch.nn.Module):
    def __init__(self, Nx, Ny, hidden=64, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = torch.nn.Sequential(
            # Input: N x 33x33
            torch.nn.Conv2d(1, 4, 3, 1, 1),
            torch.nn.Conv2d(4, 4, 3, 2, 1),
            torch.nn.LeakyReLU(.1),
            # N x 4x17x17
            torch.nn.Conv2d(4, 8, 3, 2, 1),
            torch.nn.LeakyReLU(.1),
            # N x 8x9x9
            torch.nn.Conv2d(8, 16, 3, 2, 0),
            torch.nn.LeakyReLU(.1),
            # N x 8x4x4
            torch.nn.Flatten(1, -1),
            # N x 128
            torch.nn.Linear(256, 128),
            torch.nn.LeakyReLU(.1),
            # N x 32
            torch.nn.Linear(128, hidden),
            torch.nn.LeakyReLU(.1)
        )

        self.decoder = torch.nn.Sequential(
            # Input: N x (Nx+Ny)
            torch.nn.Linear(hidden, 128),
            torch.nn.LeakyReLU(.1),
            # N x 32
            torch.nn.Linear(128, 256),
            torch.nn.LeakyReLU(.1),
            # N x 64
            torch.nn.Unflatten(-1, (16, 4, 4)),
            # N x 8x4x4
            torch.nn.ConvTranspose2d(16, 8, 3, 1, 1),
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
        return self.decoder(self.encoder(x)).squeeze()


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
y_grid, x_grid = np.meshgrid(np.linspace(0, 1, 33), np.linspace(0, 1, 33), indexing='ij')
x_sin_list = torch.tensor(np.array([np.sin(np.pi * (i + 1) * x_grid) for i in range(Nx)]), dtype=torch.float32)
y_sin_list = torch.tensor(np.array([np.sin(np.pi * (i + 1) * y_grid) for i in range(Nx)]), dtype=torch.float32)

for epoch in trange(epochs):
    for coefficients, label in loader:
        rhs = (torch.tensordot(coefficients[:, :Nx], x_sin_list, dims=1) * torch.tensordot(coefficients[:, Nx:],
                                                                                           y_sin_list,
                                                                                           dims=1)).unsqueeze(1)
        res = model(rhs)
        loss = loss_func(res, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

j = 2
plt.subplot(2, 2, 1)
plt.imshow(rhs[j].squeeze().detach())
plt.subplot(2, 2, 2)
plt.imshow(label[j].squeeze().detach())
plt.subplot(2, 2, 3)
plt.imshow(res[j].squeeze().detach())
plt.subplot(2, 2, 4)
plt.imshow((res[j] - label[j]).detach())

plt.show()
test_data, test_labels = get_dataset(Nx, Ny, '230415182814')[:]
print(f"Test loss: {loss_func(test_labels, model(test_data)).item()}")

print('done')
