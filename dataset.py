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
    H = torch.zeros((len(filenames), 1, 256, 256))
    A = torch.zeros((len(filenames), 1, 256, 256))
    for i, fn in zip(trange(len(filenames)), filenames):
        H[i] = torch.tensor(read_vtk(fn, False, 'u000', indexing='ij').reshape((1, 256, 256)), dtype=torch.float)
        A[i] = torch.tensor(read_vtk(fn, False, 'u001', indexing='ij').reshape((1, 256, 256)), dtype=torch.float)
    return H, A


class BenchData(Dataset):
    def __init__(self, basedir: str, steps: List[int], chunk_size: int = 5, overlap: int = 1, dev: torch.device = 'cpu'):
        self.velocities = read_velocities([f"{basedir}v.{n:05}.vtk" for n in steps]).to(dev)
        self.data = self.velocities[:-1] * 1e4
        self.label = self.velocities[1:] * 1e4 - self.data

        self.H, self.A = read_HA([f"{basedir}dgh.{n:05}.vtk" for n in steps[:-1]])
        self.H = self.H.to(dev)
        self.A = self.A.to(dev)

        starttime = 343.6 * 1000 / 60 / 60 / 24
        dt = 2 * 1000 / 60 / 60 / 24
        t = torch.tensor([t * dt + starttime for t in steps[:-1]], dtype=torch.float).reshape(-1, 1, 1)
        midpoint = 50 + 50 * t
        alpha = pi * 2 / 5
        x, y = torch.meshgrid(torch.linspace(0, 512, 257), torch.linspace(0, 512, 257), indexing='ij')
        x = x.reshape(1, 257, 257)
        y = y.reshape(1, 257, 257)
        r = torch.sqrt(torch.square(midpoint - x) + torch.square(midpoint - y))
        s = 1 / 50 * torch.exp(-r / 100)
        ws = torch.tanh(t * (8 - t) / 2)
        self.v_a = torch.empty_like(self.data)
        self.v_a[:, 0, :, :] = (cos(alpha) * (x - midpoint) + sin(alpha) * (y - midpoint)) * s * ws * 15  # m/s
        self.v_a[:, 1, :, :] = (-sin(alpha) * (x - midpoint) + cos(alpha) * (y - midpoint)) * s * ws * 15
        self.v_a = -self.v_a * 1e-1  # correct scaling should be *1e-3 and then *1e4

        self.v_o = torch.empty_like(self.data)
        self.v_o[:, 0] = .01 * (y / 250 - 1) * 1e-3
        self.v_o[:, 1] = .01 * (1 - x / 250) * 1e-3

        self.data_t, self.H_t, self.A_t, self.v_a_t, self.v_o_t, self.border_chunks, self.label_t = transform_data(
            self.data, self.H, self.A, self.v_a, self.v_o, self.label, chunk_size, overlap)

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


def get_patches(im: torch.Tensor, patch_size: int, overlap: int = 1):
    patches = im.unfold(2, patch_size, patch_size - overlap)
    patches = patches.unfold(3, patch_size, patch_size - overlap)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, im.shape[1], patch_size, patch_size)
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


def transform_data(data, H, A, v_a, v_o, label, chunk_size: int = 5, overlap: int = 1):
    if data.dim() == 3:
        data = data[None]
        H = H[None]
        A = A[None]
        v_a = v_a[None]
        v_o = v_o[None]
        label = label[None]

    if overlap > 2:
        padding = int((overlap - 2) / 2)
        data = torch.nn.functional.pad(data, (padding, padding, padding, padding))
        H = torch.nn.functional.pad(H, (padding, padding, padding, padding))
        A = torch.nn.functional.pad(A, (padding, padding, padding, padding))
        v_a = torch.nn.functional.pad(v_a, (padding, padding, padding, padding))
        v_o = torch.nn.functional.pad(v_o, (padding, padding, padding, padding))

    assert chunk_size > overlap, "chunk_size needs to be larger than overlap"
    assert overlap > 0, "overlap needs to be greater than 0"
    assert (data.shape[2] - overlap) % (chunk_size - overlap) == 0, "Height cannot be divided using chunk_size and overlap"
    assert (data.shape[3] - overlap) % (chunk_size - overlap) == 0, "Width cannot be divided using chunk_size and overlap"
    n_chunks = int((data.shape[2] - overlap) / (chunk_size - overlap))

    data = get_patches(data, chunk_size, overlap)
    v_a = get_patches(v_a, chunk_size, overlap)
    v_o = get_patches(v_o, chunk_size, overlap)

    if overlap % 2 == 0:
        # padding = int(overlap / 2)
        label = get_patches(label[:, :, 1:-1, 1:-1], chunk_size - overlap, 0)
    else:
        label = get_patches(label, chunk_size, overlap)

    H = get_patches(H, chunk_size - 1, overlap - 1)
    A = get_patches(A, chunk_size - 1, overlap - 1)

    border_chunks = torch.arange(data.shape[0], device=data.device).reshape(-1, n_chunks, n_chunks)
    border_chunks = (border_chunks % n_chunks == 0) + (border_chunks % n_chunks == (n_chunks - 1))
    border_chunks = border_chunks + border_chunks.transpose(1, 2)
    border_chunks = border_chunks.reshape(-1, 1, 1, 1).to(torch.float32)
    return data, H, A, v_a, v_o, border_chunks, label
