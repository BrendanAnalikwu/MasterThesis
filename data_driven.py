from math import cos, sin
from typing import List
import numpy as np
import torch
from numpy import pi
from tqdm import trange
import matplotlib.pyplot as plt

from generate_data import read_vtk
from loss import loss_func
from surrogate_net import SurrogateNet

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


def read_velocities(filenames: List[str]):
    res = torch.zeros((len(filenames), 2, 257, 257))
    for i, fn in zip(trange(len(filenames)), filenames):
        res[i, 0] = torch.tensor(read_vtk(fn, True, 'u000', indexing='ij').reshape((257, 257)), dtype=torch.float)
        res[i, 1] = torch.tensor(read_vtk(fn, True, 'u001', indexing='ij').reshape((257, 257)), dtype=torch.float)
    return res


def read_HA(filenames: List[str]):
    H = torch.zeros((len(filenames), 256, 256))
    A = torch.zeros((len(filenames), 256, 256))
    for i, fn in zip(trange(len(filenames)), filenames):
        H[i] = torch.tensor(read_vtk(fn, False, 'u000', indexing='ij').reshape((1, 256, 256)), dtype=torch.float)
        A[i] = torch.tensor(read_vtk(fn, False, 'u001', indexing='ij').reshape((1, 256, 256)), dtype=torch.float)
    return H, A


model = SurrogateNet().to(dev)

velocities = read_velocities([f"C:\\Users\\Brend\\Thesis\\GAS\\seaice\\benchmark\\Results8\\v.{n:05}.vtk" for n in range(1, 97)]).to(dev)
data = velocities[:95] * 1e4
label = velocities[1:] * 1e4 - data

H, A = read_HA([f"C:\\Users\\Brend\\Thesis\\GAS\\seaice\\benchmark\\Results8\\dgh.{n:05}.vtk" for n in range(1, 96)])
H = H.to(dev)
A = A.to(dev)

starttime = 343.6 * 1000 / 60 / 60 / 24
dt = 2 * 1000 / 60 / 60 / 24
t = torch.tensor([t * dt + starttime for t in range(1, 96)], dtype=torch.float).reshape(-1, 1, 1)
midpoint = 50 + 50 * t
alpha = pi * 2 / 5
x, y = torch.meshgrid(torch.linspace(0, 512, 257), torch.linspace(0, 512, 257), indexing='ij')
x = x.reshape(1, 257, 257)
y = y.reshape(1, 257, 257)
r = torch.sqrt(torch.square(midpoint - x) + torch.square(midpoint - y))
s = 1 / 50 * torch.exp(-r / 100)
ws = torch.tanh(t * (8 - t) / 2)
v_a = torch.empty(95, 2, 257, 257)
v_a[:, 0, :, :] = (cos(alpha) * (x - midpoint) + sin(alpha) * (y - midpoint)) * s * ws * 15  # m/s
v_a[:, 1, :, :] = (-sin(alpha) * (x - midpoint) + cos(alpha) * (y - midpoint)) * s * ws * 15
v_a = -v_a * 1e-1  # correct scaling should be *1e-3 and then *1e4
v_a = v_a.to(dev)

v_o = torch.empty(1, 2, 257, 257, device=dev)
v_o[:, 0] = .01 * (y/250 - 1) * 1e-3
v_o[:, 1] = .01 * (1 - x/250) * 1e-3

criterion = torch.nn.MSELoss().to(dev)
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
losses = []
PINN_losses = []

loss_func(label * 1e-4, H, A, data * 1e-4, v_a * 1e-2, v_o, C_r, C_a, C_o, T, e_2, C, f_c, dx, 1.)

pbar = trange(10000)
for i in pbar:
    # if i == 2000:
    #     for g in optim.param_groups:
    #         g['lr'] = 1e-5
    idx = np.array_split(np.random.permutation(range(len(data))), 5)
    for mb in idx:
        output = model(data[mb], H[mb], A[mb], v_a[mb])
        loss = criterion(output, label[mb])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)
        optim.step()
        losses.append(loss.item())

        with torch.no_grad():
            PINN_losses.append(
                loss_func(output * 1e-4, H[mb], A[mb], data[mb] * 1e-4, v_a[mb] * 1e-2, v_o, C_r, C_a, C_o, T, e_2,
                          C, f_c, dx, dt).item())
        torch.cuda.empty_cache()
    # pbar.set_postfix(loss=((output.detach()[0,0]-label.detach()[mb[0],0]).abs().mean()/label.detach()[mb[0],0].abs().mean()).cpu())
    pbar.set_postfix(loss=loss.item(), PINN=PINN_losses[-1])

print('done')
