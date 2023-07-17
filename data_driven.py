from math import cos, sin
from typing import List
import numpy as np
import torch
from numpy import pi
from tqdm import trange
import matplotlib.pyplot as plt

from generate_data import read_vtk
from surrogate_net import SurrogateNet

dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def read_velocities(filenames: List[str]):
    res = torch.zeros((len(filenames), 2, 257, 257))
    for i, fn in zip(trange(len(filenames)), filenames):
        res[i, 0] = torch.tensor(read_vtk(fn, True, 'u000').reshape((257, 257)), dtype=torch.float)
        res[i, 1] = torch.tensor(read_vtk(fn, True, 'u001').reshape((257, 257)), dtype=torch.float)
    return res


def read_HA(filenames: List[str]):
    H = torch.zeros((len(filenames), 256, 256))
    A = torch.zeros((len(filenames), 256, 256))
    for i, fn in zip(trange(len(filenames)), filenames):
        H[i] = torch.tensor(read_vtk(fn, False, 'u000').reshape((1, 256, 256)), dtype=torch.float)
        A[i] = torch.tensor(read_vtk(fn, False, 'u001').reshape((1, 256, 256)), dtype=torch.float)
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
x, y = torch.meshgrid(torch.linspace(0, 512, 257), torch.linspace(0, 512, 257), indexing='xy')
x = x.reshape(1, 257, 257)
y = y.reshape(1, 257, 257)
r = torch.sqrt(torch.square(midpoint - x) + torch.square(midpoint - y))
s = 1 / 50 * torch.exp(-r / 100)
ws = torch.tanh(t * (8 - t) / 2)
v_a = torch.empty(95, 2, 257, 257)
v_a[:, 0, :, :] = (cos(alpha) * (x - midpoint) + sin(alpha) * (y - midpoint)) * s * ws * 15
v_a[:, 1, :, :] = (-sin(alpha) * (x - midpoint) + cos(alpha) * (y - midpoint)) * s * ws * 15
v_a = -v_a * 1e-1
v_a = v_a.to(dev)

criterion = torch.nn.MSELoss().to(dev)
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
losses = []

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
    # pbar.set_postfix(loss=((output.detach()[0,0]-label.detach()[mb[0],0]).abs().mean()/label.detach()[mb[0],0].abs().mean()).cpu())
    pbar.set_postfix(loss=loss.item())

print('done')
