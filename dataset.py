from math import cos, sin, pi, sqrt
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
from torchvision.utils import make_grid
from tqdm import trange

from generate_data import read_vtk


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
        H[i] = torch.tensor(read_vtk(fn, False, 'u000', indexing='ij').reshape((1, 1, 256, 256)), dtype=torch.float)
        A[i] = torch.tensor(read_vtk(fn, False, 'u001', indexing='ij').reshape((1, 1, 256, 256)), dtype=torch.float)
    return H, A


class BenchData(Dataset):
    def __init__(self, basedir: str, steps: List[int], dev: torch.device = 'cpu'):
        self.velocities = read_velocities([f"{basedir}v.{n:05}.vtk" for n in steps]).to(dev)
        self.data = self.velocities[:-1] * 1e4
        self.label = self.velocities[1:] * 1e4 - self.data

        self.H, self.A = read_HA([f"{basedir}dgh.{n:05}.vtk" for n in steps[:-1]])
        self.H = self.H.to(dev)
        self.A = self.A.to(dev)

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
        self.v_a = torch.empty(95, 2, 257, 257, device=dev)
        self.v_a[:, 0, :, :] = (cos(alpha) * (x - midpoint) + sin(alpha) * (y - midpoint)) * s * ws * 15  # m/s
        self.v_a[:, 1, :, :] = (-sin(alpha) * (x - midpoint) + cos(alpha) * (y - midpoint)) * s * ws * 15
        self.v_a = -self.v_a * 1e-1  # correct scaling should be *1e-3 and then *1e4

        self.v_o = torch.empty(95, 2, 257, 257, device=dev)
        self.v_o[:, 0] = .01 * (y / 250 - 1) * 1e-3
        self.v_o[:, 1] = .01 * (1 - x / 250) * 1e-3

        self.data_t, self.H_t, self.A_t, self.v_a_t, self.v_o_t, self.border_chunks, self.label_t = transform_data(
            self.data, self.H, self.A, self.v_a, self.v_o, self.label)

    def __getitem__(self, index) -> T_co:
        return (self.data_t[index], self.H_t[index], self.A_t[index], self.v_a_t[index],
                self.v_o_t[index], self.border_chunks[index], self.label_t[index])

    def retrieve(self, index) -> T_co:
        return (self.data[index], self.H[index], self.A[index], self.v_a[index],
                self.v_o[index], self.label[index])

    def __len__(self) -> int:
        return len(self.data_t)


def random_crop_collate(crop_size: int):
    def func(batch):
        crop_x = np.random.randint(256 - crop_size + 1, size=len(batch))
        crop_y = np.random.randint(256 - crop_size + 1, size=len(batch))
        data, H, A, v_a, v_o, label = zip(*batch)
        data = torch.stack(
            [item[:, crop_x[i]:(crop_x[i] + crop_size + 1), crop_y[i]:(crop_y[i] + crop_size + 1)] for i, item in
             enumerate(data)], dim=0)
        H = torch.stack(
            [item[crop_x[i]:(crop_x[i] + crop_size), crop_y[i]:(crop_y[i] + crop_size)] for i, item in
             enumerate(H)])
        A = torch.stack(
            [item[crop_x[i]:(crop_x[i] + crop_size), crop_y[i]:(crop_y[i] + crop_size)] for i, item in
             enumerate(A)])
        v_a = torch.stack(
            [item[:, crop_x[i]:(crop_x[i] + crop_size + 1), crop_y[i]:(crop_y[i] + crop_size + 1)] for i, item in
             enumerate(v_a)], dim=0)
        v_o = torch.stack(
            [item[:, crop_x[i]:(crop_x[i] + crop_size + 1), crop_y[i]:(crop_y[i] + crop_size + 1)] for i, item in
             enumerate(v_o)], dim=0)
        label = torch.stack(
            [item[:, crop_x[i]:(crop_x[i] + crop_size + 1), crop_y[i]:(crop_y[i] + crop_size + 1)] for i, item in
             enumerate(label)], dim=0)
        return data, H, A, v_a, v_o, label
    return func


def get_patches(im: torch.Tensor, patch_size: int):
    patches = im.unfold(2, patch_size+1, patch_size)
    patches = patches.unfold(3, patch_size+1, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, 2, patch_size+1, patch_size+1)
    return patches


def stitch(im: torch.Tensor, batch_size: int):
    assert len(im.shape) == 4, "Tensor is not 4D"
    assert im.shape[0] % batch_size == 0, "First dimension not divisible by batch_size"
    patch_size = im.shape[-1] - 1
    patches_1d = sqrt(im.shape[0] / batch_size)
    assert patches_1d.is_integer(), "Wrong dimensions"

    im = im.reshape(batch_size, -1, 2, patch_size + 1, patch_size + 1)
    im = torch.stack([make_grid(im[i], int(patches_1d), padding=0) for i in range(im.shape[0])], dim=0)
    mask = np.arange(patches_1d * 5) % 5 != 4
    mask[-1] = True
    im = im[:, :, mask][:, :, :, mask]
    return im


def transform_data(data, H, A, v_a, v_o, label, chunk_size: int = 4, overlap: int = 1):
    data = get_patches(data, chunk_size + overlap, overlap)
    v_a = get_patches(v_a, chunk_size + overlap, overlap)
    v_o = get_patches(v_o, chunk_size + overlap, overlap)
    label = get_patches(label, chunk_size + overlap, overlap)

    H = torch.stack(torch.stack(H.split(chunk_size, 2), 1).split(chunk_size, 4), 2).reshape(-1, chunk_size, chunk_size)
    A = torch.stack(torch.stack(A.split(chunk_size, 2), 1).split(chunk_size, 4), 2).reshape(-1, chunk_size, chunk_size)

    border_chunks = torch.arange(data.shape[0], device=data.device).reshape(-1, 64, 64)
    border_chunks = (border_chunks % 64 == 0) + (border_chunks % 64 == 63)
    border_chunks = border_chunks + border_chunks.transpose(1, 2)
    border_chunks = border_chunks.reshape(-1, 1)
    return data, H, A, v_a, v_o, border_chunks, label
