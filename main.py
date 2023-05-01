import glob
from math import ceil
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

        self.encoder1 = torch.nn.Sequential(
            # Input: N x 1x33x33
            torch.nn.Conv2d(1, 4, 3, 1, 1),
            torch.nn.LeakyReLU(.1),
            # N x 4x33x33
            torch.nn.Conv2d(4, 4, 3, 2, 1),
            torch.nn.LeakyReLU(.1))
        self.encoder2 = torch.nn.Sequential(
            # N x 4x17x17
            torch.nn.Conv2d(4, 8, 3, 2, 1),
            torch.nn.LeakyReLU(.1),
            # N x 8x9x9
            torch.nn.Conv2d(8, 16, 3, 2, 0),
            torch.nn.LeakyReLU(.1),
            # N x 16x4x4
            torch.nn.Flatten(1, -1),
            # N x 256
            torch.nn.Linear(256, 128),
            torch.nn.LeakyReLU(.1),
            # N x 32
            torch.nn.Linear(128, hidden),
            torch.nn.LeakyReLU(.1)
        )

        self.decoder1 = torch.nn.Sequential(
            # Input: N x (Nx+Ny)
            torch.nn.Linear(hidden, 128),
            torch.nn.LeakyReLU(.1),
            # N x 32
            torch.nn.Linear(128, 256),
            torch.nn.LeakyReLU(.1),
            # N x 64
            torch.nn.Unflatten(-1, (16, 4, 4)),
            # N x 8x4x4
            # torch.nn.ConvTranspose2d(16, 8, 3, 1, 1),
            torch.nn.ConvTranspose2d(16, 8, 3, 2, 0),
            torch.nn.ReLU(),
            # N x 8x9x9
            # torch.nn.ConvTranspose2d(8, 4, 3, 1, 1),
            torch.nn.ConvTranspose2d(8, 4, 3, 2, 1),
            torch.nn.ReLU())
        self.decoder2 = torch.nn.Sequential(
            # N x 4x17x17
            # torch.nn.ConvTranspose2d(4, 4, 3, 1, 1),
            torch.nn.ConvTranspose2d(4, 4, 3, 2, 1),
            torch.nn.ReLU(),
            # N x 4x33x33
            # torch.nn.ConvTranspose2d(4, 4, 3, 1, 1),
            torch.nn.ConvTranspose2d(4, 1, 3, 1, 1),
            # N x 1x33x33
            torch.nn.Tanh()
        )

    def forward(self, x: torch.Tensor):
        x_skip = self.encoder1(x)
        output = self.decoder2(x_skip + self.decoder1(self.encoder2(x_skip)))
        return output.squeeze()


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

test_losses = []
test_data, test_labels = get_dataset(Nx, Ny, '230415182814')[:]
test_rhs = (torch.tensordot(test_data[:, :Nx], x_sin_list, dims=1)
            * torch.tensordot(test_data[:, Nx:], y_sin_list, dims=1)).unsqueeze(1)
test_losses.append(loss_func(model(test_rhs), test_labels).item())

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

    test_losses.append(loss_func(model(test_rhs), test_labels).item())


def plot_sample(num_sol: torch.Tensor, net_res: torch.Tensor, righths: torch.Tensor, j: int = 0):
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(num_sol[j].squeeze().detach())
    plt.title('Numerical solution')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 2, 2)
    plt.imshow(net_res[j].squeeze().detach())
    plt.title('Network result')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 2, 3)
    plt.imshow(righths[j].squeeze().detach())
    plt.title('RHS')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 2, 4)
    plt.imshow((net_res[j] - num_sol[j]).detach())
    plt.title('Error')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()


def plot_losses(train: list[float], test: list[float], batch_size: int = 32, train_size: int = 30000):
    plt.figure()
    plt.semilogy(train)
    plt.semilogy([i * ceil(train_size / batch_size) for i in range(len(test))], test)
    plt.title("Loss (MSE)")
    plt.legend(['Training data', 'Test data'])
    plt.tight_layout()
    plt.show()


plot_sample(label, res, rhs, 0)
plot_losses(losses, test_losses)

print(f"Test loss: {loss_func(test_labels, model(test_rhs)).item()}")

print('done')
