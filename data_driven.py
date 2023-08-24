from math import cos, sin
from typing import List
import numpy as np
import torch
from numpy import pi
from torch.utils.data import DataLoader
from tqdm import trange
import matplotlib.pyplot as plt

from dataset import BenchData, random_crop_collate
from generate_data import read_vtk
from loss import loss_func
from surrogate_net import SurrogateNet, PatchNet

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


model = PatchNet().to(dev)

# velocities = read_velocities([f"C:\\Users\\Brend\\Thesis\\GAS\\seaice\\benchmark\\Results8\\v.{n:05}.vtk" for n in range(1, 97)]).to(dev)
# data = velocities[:95] * 1e4
# label = velocities[1:] * 1e4 - data
#
# H, A = read_HA([f"C:\\Users\\Brend\\Thesis\\GAS\\seaice\\benchmark\\Results8\\dgh.{n:05}.vtk" for n in range(1, 96)])
# H = H.to(dev)
# A = A.to(dev)
#
# starttime = 343.6 * 1000 / 60 / 60 / 24
# dt = 2 * 1000 / 60 / 60 / 24
# t = torch.tensor([t * dt + starttime for t in range(1, 96)], dtype=torch.float).reshape(-1, 1, 1)
# midpoint = 50 + 50 * t
# alpha = pi * 2 / 5
# x, y = torch.meshgrid(torch.linspace(0, 512, 257), torch.linspace(0, 512, 257), indexing='ij')
# x = x.reshape(1, 257, 257)
# y = y.reshape(1, 257, 257)
# r = torch.sqrt(torch.square(midpoint - x) + torch.square(midpoint - y))
# s = 1 / 50 * torch.exp(-r / 100)
# ws = torch.tanh(t * (8 - t) / 2)
# v_a = torch.empty(95, 2, 257, 257)
# v_a[:, 0, :, :] = (cos(alpha) * (x - midpoint) + sin(alpha) * (y - midpoint)) * s * ws * 15  # m/s
# v_a[:, 1, :, :] = (-sin(alpha) * (x - midpoint) + cos(alpha) * (y - midpoint)) * s * ws * 15
# v_a = -v_a * 1e-1  # correct scaling should be *1e-3 and then *1e4
# v_a = v_a.to(dev)
#
# v_o = torch.empty(95, 2, 257, 257, device=dev)
# v_o[:, 0] = .01 * (y/250 - 1) * 1e-3
# v_o[:, 1] = .01 * (1 - x/250) * 1e-3

dataset = BenchData("C:\\Users\\Brend\\Thesis\\GAS\\seaice\\benchmark\\Results8\\", list(range(1, 97)), dev)
dataloader = DataLoader(dataset, batch_size=10000, shuffle=True)  # , collate_fn=random_crop_collate(crop_size=4))

criterion = torch.nn.MSELoss().to(dev)
optim = torch.optim.Adam(model.parameters(), lr=1e-5)
losses = []
PINN_losses = []

# loss_func(label * 1e-4, H, A, data * 1e-4, v_a * 1e-2, v_o, C_r, C_a, C_o, T, e_2, C, f_c, dx, 1.)

crop_size = 4
pbar = trange(int(2e3))
for i in pbar:
    # if i == 200:
    #     for g in optim.param_groups:
    #         g['lr'] = 1e-5
    for (data, H, A, v_a, v_o, label) in dataloader:
        output = model(data, H, A, v_a, v_o)
        # if i < 200:
        loss = criterion(output, label)
        losses.append(loss.item())
        # with torch.no_grad():
        #     PINN_losses.append(
        #         loss_func(output * 1e-4, H[mb], A[mb], data[mb] * 1e-4, v_a[mb] * 1e-2, v_o, C_r, C_a, C_o, T, e_2,
        #                   C, f_c, dx, dt).item())
        # else:
        #     loss = loss_func(output * 1e-4, H[mb], A[mb], data[mb] * 1e-4, v_a[mb] * 1e-2, v_o, C_r, C_a, C_o, T, e_2,
        #                      C, f_c, dx, dt)
        #     PINN_losses.append(loss.item())
        #     with torch.no_grad():
        #         losses.append(criterion(output, label[mb]).item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optim.step()
        torch.cuda.empty_cache()
    # pbar.set_postfix(loss=((output.detach()[0,0]-label.detach()[mb[0],0]).abs().mean()/label.detach()[mb[0],0].abs().mean()).cpu())
    pbar.set_postfix(loss=losses[-1])  # , PINN=PINN_losses[-1])

print('done')
