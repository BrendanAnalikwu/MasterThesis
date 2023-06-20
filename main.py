import torch
import torch.utils.data
from tqdm import trange
from loss import loss_func, vector, form
from surrogate_net import SurrogateNet
import matplotlib.pyplot as plt


def advect(v: torch.Tensor, H_old: torch.tensor, dt: float, dx: float):
    """
    Perform advection step on B-grid.

    Parameters
    ----------
    v : torch.Tensor
        Ice velocity tensor of shape (N, 2, H+1, W+1)
    H_old : torch.Tensor
        Quantity to advect as tensor with shape (N, H, W)
    dt : float
        Time step
    dx : float
        Grid spacing

    Returns
    -------
    torch.Tensor
        Advected quantity

    """

    dH = torch.zeros_like(H_old)

    # x-direction
    v_x = (v[:, 0, 1:, 1:-1] + v[:, 0, :-1, 1:-1]) / 2.
    dH[:, :, 1:] += torch.nn.functional.relu(v_x) * H_old[:, :, :-1] * dt / dx
    dH[:, :, :-1] -= torch.nn.functional.relu(v_x) * H_old[:, :, :-1] * dt / dx
    dH[:, :, :-1] += torch.nn.functional.relu(-1 * v_x) * H_old[:, :, 1:] * dt / dx
    dH[:, :, 1:] -= torch.nn.functional.relu(-1 * v_x) * H_old[:, :, 1:] * dt / dx

    # y-direction
    v_y = (v[:, 0, 1:-1, 1:] + v[:, 0, 1:-1, :-1]) / 2.
    dH[:, 1:, :] += torch.nn.functional.relu(-1 * v_y) * H_old[:, :-1, :] * dt / dx
    dH[:, :-1, :] -= torch.nn.functional.relu(-1 * v_y) * H_old[:, :-1, :] * dt / dx
    dH[:, :-1, :] += torch.nn.functional.relu(v_y) * H_old[:, 1:, :] * dt / dx
    dH[:, 1:, :] -= torch.nn.functional.relu(v_y) * H_old[:, 1:, :] * dt / dx

    return H_old + dH


# Constants
L = 1e4
T = 1e3
G = 1.
C_o = 1026 * 5.5e-3 * L / (900 * G)
C_a = 1.3 * 1.2e-3 * L / (900 * G)
C_r = 27.5e3 * T ** 2 / (2 * 900 * L ** 2)
e_2 = .5
C = 20
f_c = 1.46e-4
dx = 512e3 / 128 / L  # 4km
dt = 1e3 / T  # 1000s

# Initial conditions
N = 4
x, y = torch.meshgrid(torch.linspace(0, 1, 129), torch.linspace(0, 1, 129), indexing='xy')
x_h, y_h = torch.meshgrid(torch.linspace(0, 1 - 1 / 128., 128), torch.linspace(0, 1 - 1 / 128., 128), indexing='xy')
H0 = .3 * torch.ones(N, 128, 128) + .005 * (torch.sin(x_h * 256) + torch.sin(y_h * 256))
A0 = torch.ones(N, 128, 128)
v0 = torch.zeros(N, 2, 129, 129)
v0[:, :, 1:-1, 1:-1] = .001 * torch.ones(N, 2, 127, 127)
v0[:2, 0, :64, :] = -v0[:2, 0, :64, :]
v0[2:, 0, :, :64] = -v0[2:, 0, :, :64]

# Setup NN
model = SurrogateNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# v, H, A = v0, H0, A0
v_a = torch.ones_like(v0) * (20 * T / L)
v_a[::2, 0, :64, :] = -v_a[::2, 0, :64, :]
losses = []

H = advect(v0, H0, dt, dx)
H.clamp_(0., None)
A = advect(v0, A0, dt, dx)
A.clamp_(0., 1.)

for i in trange(10000):
    # v_old = v.detach()
    v = model(v0, H0, A0, v_a)
    loss = loss_func(v, H, A, v0, v_a, C_r, C_a, T, e_2, C, f_c, dx, dt)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    if i == 500:
        for g in optimizer.param_groups:
            g['lr'] = 1e-4
    if i == 3000:
        for g in optimizer.param_groups:
            g['lr'] = 1e-5
    optimizer.step()
    losses.append(loss.item())

print('done')
