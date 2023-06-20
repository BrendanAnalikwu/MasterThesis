import glob
from math import ceil
from typing import Tuple
import torch
import torch.utils.data
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt


def delta(vx_x: torch.Tensor, vx_y: torch.Tensor, vy_x: torch.Tensor, vy_y: torch.Tensor, e_2: float = .25,
          dmin: float = 2e-9):
    return ((1 + e_2) * (torch.square(vx_x) + torch.square(vy_y))
            + 2 * (1 - e_2) * torch.mul(vx_x, vy_y)
            + e_2 * torch.square(vx_y + vy_x)
            + dmin ** 2)


def finite_differences(v: torch.Tensor, dx: float):
    v_x = (v[:, :, :-1, 1:] + v[:, :, 1:, 1:] - v[:, :, :-1, :-1] - v[:, :, 1:, :-1]) / dx / 2.
    v_y = (v[:, :, 1:, :-1] + v[:, :, 1:, 1:] - v[:, :, :-1, :-1] - v[:, :, :-1, 1:]) / dx / 2.
    return v_x, v_y


def internal_stress(v: torch.Tensor, H: torch.Tensor, A: torch.Tensor, dx: float, C: float, e_2: float):
    v_x, v_y = finite_differences(v, dx)
    delta_ = delta(v_x[:, 0, :, :], v_y[:, 0, :, :], v_x[:, 1, :, :], v_y[:, 1, :, :])
    trace = v_x[:, 0, :, :] + v_y[:, 1, :, :]
    s_xx = H * torch.exp(-C * (1 - A)) * ((2 * e_2 * v_x[:, 0, :, :] + (1 - e_2) * trace) / delta_ - 1)
    s_yy = H * torch.exp(-C * (1 - A)) * ((2 * e_2 * v_y[:, 1, :, :] + (1 - e_2) * trace) / delta_ - 1)
    s_xy = H * torch.exp(-C * (1 - A)) * e_2 * (v_x[:, 1, :, :] + v_y[:, 0, :, :]) / delta_
    return s_xx, s_yy, s_xy


def form(v: torch.Tensor, H: torch.Tensor, A: torch.Tensor, dx: float, C: float, e_2: float, dt: float, T: float,
         f_c: float, C_r: float):
    s_xx, s_yy, s_xy = internal_stress(v, H, A, dx, C, e_2)

    # v_filter = dx ** 2 / 36 * torch.tensor([[1, 4, 1], [4, 16, 4], [1, 4, 1]], dtype=torch.float32)
    v_filter = dx ** 2 / 36 * torch.tensor([[2, 4], [1, 2]], dtype=torch.float32)[None, :, :]
    v_filter = torch.stack((v_filter, v_filter))

    A = (torch.mul(torch.nn.functional.conv2d(v[:, :, 1:, :-1], v_filter, groups=2), H[:, None, 1:, :-1])
         + torch.mul(torch.nn.functional.conv2d(v[:, :, 1:, 1:], torch.flip(v_filter, [3]), groups=2),
                     H[:, None, 1:, 1:])
         + torch.mul(torch.nn.functional.conv2d(v[:, :, :-1, :-1], torch.flip(v_filter, [2]), groups=2),
                     H[:, None, :-1, :-1])
         + torch.mul(torch.nn.functional.conv2d(v[:, :, :-1, 1:], torch.flip(v_filter, [2, 3]), groups=2),
                     H[:, None, :-1, 1:])) / dt

    s_filter = dx ** 2 / 3 * torch.tensor([[1, 1], [1, 1]], dtype=torch.float32)[None, None, :, :]
    A[:, 0:1, :, :] -= C_r * torch.nn.functional.conv2d((s_xx + s_xy)[:, None, ...], s_filter)
    A[:, 1:2, :, :] -= C_r * torch.nn.functional.conv2d((s_xy + s_yy)[:, None, ...], s_filter)

    return A


def vector(H: torch.Tensor, A: torch.Tensor, v_old: torch.Tensor, v_a: torch.Tensor, C_a: float,
           dx: float, dt: float):
    v_filter = dx ** 2 / 36 * torch.tensor([[2, 4], [1, 2]], dtype=torch.float32)[None, :, :]
    v_filter = torch.stack((v_filter, v_filter))

    F = (torch.mul(torch.nn.functional.conv2d(v_old[:, :, 1:, :-1], v_filter, groups=2), H[:, None, 1:, :-1])
         + torch.mul(torch.nn.functional.conv2d(v_old[:, :, 1:, 1:], torch.flip(v_filter, [3]), groups=2),
                     H[:, None, 1:, 1:])
         + torch.mul(torch.nn.functional.conv2d(v_old[:, :, :-1, :-1], torch.flip(v_filter, [2]), groups=2),
                     H[:, None, :-1, :-1])
         + torch.mul(torch.nn.functional.conv2d(v_old[:, :, :-1, 1:], torch.flip(v_filter, [2, 3]), groups=2),
                     H[:, None, :-1, 1:])) / dt

    v_a_filter = dx ** 2 / 36 * torch.tensor([[2, 4], [1, 2]], dtype=torch.float32)[None, :, :]
    v_a_filter = torch.stack((v_a_filter, v_a_filter))

    t_a = C_a * torch.mul(torch.linalg.norm(v_a, dim=0, keepdim=True), v_a)
    F += (torch.mul(torch.nn.functional.conv2d(t_a[:, :, 1:, :-1], v_a_filter, groups=2), A[:, None, 1:, :-1])
          + torch.mul(torch.nn.functional.conv2d(t_a[:, :, 1:, 1:], torch.flip(v_a_filter, [3]), groups=2),
                      A[:, None, 1:, 1:])
          + torch.mul(torch.nn.functional.conv2d(t_a[:, :, :-1, :-1], torch.flip(v_a_filter, [2]), groups=2),
                      A[:, None, :-1, :-1])
          + torch.mul(torch.nn.functional.conv2d(t_a[:, :, :-1, 1:], torch.flip(v_a_filter, [2, 3]), groups=2),
                      A[:, None, :-1, 1:])) / dt

    return F


def loss_func(v: torch.Tensor, H: torch.Tensor, A: torch.Tensor, v_old: torch.Tensor, v_a: torch.Tensor, C_r, C_a, T,
              e_2, C, f_c, dx, dt):
    return (1e6 * torch.sum(torch.pow(vector(H, A, v_old, v_a, C_a, dx, dt)
                                      - form(v, H, A, dx, C, e_2, dt, T, f_c, C_r), 2))
            + torch.sum(torch.pow(v[:, :, 0, :], 2)) + torch.sum(torch.pow(v[:, :, -1, :], 2))
            + torch.sum(torch.pow(v[:, :, 0, 1:-1], 2)) + torch.sum(torch.pow(v[:, :, -1, 1:-1], 2))) * 1e6


class SurrogateNet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layer1 = torch.nn.Conv2d(4, 4, 2, 1, 0)
        self.layer2 = torch.nn.Sequential(
            # 128
            torch.nn.Conv2d(6, 8, 3, 1, 0),
            torch.nn.LeakyReLU(.1),
            # 126
            torch.nn.Conv2d(8, 8, 3, 1, 0),
            torch.nn.LeakyReLU(.1),
            # 124
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.MaxPool2d(2, 2),
            # 62
            torch.nn.Conv2d(8, 16, 3, 1, 0),
            torch.nn.LeakyReLU(.1),
            # 60
            torch.nn.Conv2d(16, 16, 3, 1, 0),
            torch.nn.LeakyReLU(.1),
            # 58
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(2, 2),
            # 29
            torch.nn.Conv2d(16, 32, 3, 1, 0),
            torch.nn.LeakyReLU(.1),
            # 27
            torch.nn.Conv2d(32, 32, 3, 2, 0),
            torch.nn.LeakyReLU(.1),
            # 13
            torch.nn.Conv2d(32, 32, 3, 1, 1),
            torch.nn.LeakyReLU(.1),
            # 13
            torch.nn.Conv2d(32, 32, 3, 2, 0),
            torch.nn.LeakyReLU(.1),
            # 6
            torch.nn.Flatten(1, -1),
            # 2304
            torch.nn.Linear(1152, 512),
            torch.nn.LeakyReLU(.1),
            torch.nn.Linear(512, 512),
            torch.nn.LeakyReLU(.1),
            torch.nn.Linear(512, 1152),
            torch.nn.Unflatten(-1, (32, 6, 6)),
            # 6
            torch.nn.ConvTranspose2d(32, 32, 3, 2, 0),
            torch.nn.ReLU(),
            # 13
            torch.nn.Conv2d(32, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 32, 3, 2, 0),
            torch.nn.ReLU(),
            # 27
            torch.nn.ConvTranspose2d(32, 16, 3, 1, 0),
            torch.nn.LeakyReLU(.1),
            # 29
            torch.nn.ConvTranspose2d(16, 16, 2, 2, 0),
            torch.nn.LeakyReLU(.1)
            # 58
        )
        self.layer5 = torch.nn.Sequential(
            # 58
            torch.nn.ConvTranspose2d(32, 16, 3, 1, 0),
            torch.nn.LeakyReLU(.1),
            # 60
            torch.nn.ConvTranspose2d(16, 16, 3, 1, 0),
            torch.nn.LeakyReLU(.1),
            # 62
            torch.nn.ConvTranspose2d(16, 8, 2, 2, 0),
            torch.nn.LeakyReLU(.1),
            # 124
        )
        self.layer6 = torch.nn.Sequential(
            # 124
            torch.nn.ConvTranspose2d(16, 8, 3, 1, 0),
            torch.nn.LeakyReLU(.1),
            # 126
            torch.nn.ConvTranspose2d(8, 8, 3, 1, 0),
            torch.nn.LeakyReLU(.1)
            # 128
        )
        self.layer7 = torch.nn.Sequential(torch.nn.ConvTranspose2d(8, 4, 2, 1, 0),
                                          torch.nn.Tanhshrink(),
                                          torch.nn.ConvTranspose2d(4, 2, 3, 1, 1),
                                          torch.nn.Tanh())

    def forward(self, v, H, A, v_a):
        x = self.layer1(torch.cat((v, v_a), 1))
        x = torch.cat((x, H[:, None, ...], A[:, None, ...]), 1)
        x1 = self.layer2(x)  # 124
        x2 = self.layer3(x1)  # 58
        x3 = self.layer4(x2)
        x4 = self.layer5(torch.cat((x3, x2), 1))
        x5 = self.layer6(torch.cat((x4, x1), 1))
        dv = self.layer7(x5)

        return v + .1 * dv  # , torch.nn.functional.relu(x5[:, 2, :, :]), torch.nn.functional.sigmoid(x5[:, 3, :, :])


def advect(v: torch.Tensor, H_old: torch.tensor):
    dH = torch.zeros_like(H_old)

    # x-direction
    v_x = (v[:, 0, 1:, 1:-1] + v[:, 0, :-1, 1:-1]) / 2.
    dH[:, :, 1:] += torch.nn.functional.relu(v_x) * H_old[:, :, :-1]
    dH[:, :, :-1] -= torch.nn.functional.relu(v_x) * H_old[:, :, :-1]
    dH[:, :, :-1] += torch.nn.functional.relu(-1 * v_x) * H_old[:, :, 1:]
    dH[:, :, 1:] -= torch.nn.functional.relu(-1 * v_x) * H_old[:, :, 1:]

    # y-direction
    v_y = (v[:, 0, 1:-1, 1:] + v[:, 0, 1:-1, :-1]) / 2.
    dH[:, 1:, :] += torch.nn.functional.relu(-1 * v_y) * H_old[:, :-1, :]
    dH[:, :-1, :] -= torch.nn.functional.relu(-1 * v_y) * H_old[:, :-1, :]
    dH[:, :-1, :] += torch.nn.functional.relu(v_y) * H_old[:, 1:, :]
    dH[:, 1:, :] -= torch.nn.functional.relu(v_y) * H_old[:, 1:, :]

    return H_old + dH


# Constants
L = 1e6
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
v0[:, :, 1:-1, 1:-1] = .01 * torch.ones(N, 2, 127, 127)
v0[:2, 0, :64, :] = -v0[:2, 0, :64, :]
v0[2:, 0, :, :64] = -v0[2:, 0, :, :64]

# Setup NN
model = SurrogateNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# v, H, A = v0, H0, A0
v_a = torch.ones_like(v0) * (20 * T / L)
v_a[::2, 0, :64, :] = -v_a[::2, 0, :64, :]
losses = []

H = advect(v0, H0)
H.clamp_(0., None)
A = advect(v0, A0)
A.clamp_(0., 1.)

for i in trange(10000):
    # v_old = v.detach()
    v = model(v0, H0, A0, v_a)
    loss = loss_func(v, H, A, v0, v_a, C_r, C_a, T, e_2, C, f_c, dx, dt)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    if i == 1000:
        for g in optimizer.param_groups:
            g['lr'] = 1e-5
    optimizer.step()
    losses.append(loss.item())

print('done')
