import os
from abc import ABC
from collections import defaultdict
from math import cos, sin, pi, sqrt, exp
from random import randint, getrandbits
from typing import List, Optional, Union, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision.utils import make_grid
from tqdm import trange, tqdm

from generate_data import read_vtk2


def read_velocities(filenames: List[str], silent=True):
    res = torch.zeros((len(filenames), 2, 257, 257))
    for i, fn in zip(trange(len(filenames), disable=silent), filenames):
        data = read_vtk2(fn, True, indexing='ij')
        res[i, 0] = torch.tensor(data[:, 0].reshape((257, 257)), dtype=torch.float)
        res[i, 1] = torch.tensor(data[:, 1].reshape((257, 257)), dtype=torch.float)
    return res


def read_HA(filenames: List[str], silent=True):
    H = torch.zeros((len(filenames), 1, 256, 256))
    A = torch.zeros((len(filenames), 1, 256, 256))
    for i, fn in zip(trange(len(filenames), disable=silent), filenames):
        data = read_vtk2(fn, False, indexing='ij')
        H[i] = torch.tensor(data[:, 0].reshape((1, 256, 256)), dtype=torch.float)
        A[i] = torch.tensor(data[:, 1].reshape((1, 256, 256)), dtype=torch.float)
    return H, A


class CoefficientDict(dict):
    def __missing__(self, key):
        return 0. if key[:2] == 'W_' or key[-4:] == '_min' or key[-4] == '_max' else []


class SeaIceDataset(Dataset, ABC):
    data: torch.Tensor
    H: torch.Tensor
    A: torch.Tensor
    v_a: torch.Tensor
    # v_o: torch.Tensor
    label: torch.Tensor
    transform = None

    def retrieve(self, index) -> T_co:
        return self.data[index], self.H[index], self.A[index], self.v_a[index], self.label[index]

    def __getitem__(self, index) -> T_co:
        sample = self.retrieve(index)
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def scale_velocity(v: torch.Tensor) -> torch.Tensor:
        return v * 1e4


class SeaIceTransform(object):
    def __call__(self, sample, rot: int = None, flip: bool = None):
        if rot is None:
            rot = randint(0, 3)  # generate a random number of rotations
        if flip is None:
            flip = bool(getrandbits(1))

        data, H, A, v_a, label = sample
        with torch.no_grad():
            return (self.transform_velocity(data, rot, flip),
                    self.transform_quantity(H, rot, flip),
                    self.transform_quantity(A, rot, flip),
                    self.transform_velocity(v_a, rot, flip),
                    self.transform_velocity(label, rot, flip))

    @staticmethod
    def transform_velocity(x: torch.Tensor, rot: int, flip: bool):
        x = x.clone()
        if rot == 1:
            x[..., 0, :, :] = x[..., 0, :, :].neg()
            x = x[..., [1, 0], :, :]
        elif rot == 2:
            x = x.neg()
        elif rot == 3:
            x[..., 1, :, :] = -1*x[..., 1, :, :]
            x = x[..., [1, 0], :, :]
        x = x.rot90(rot, [-2, -1])

        if flip:
            x[..., 1, :, :] = x[..., 1, :, :].neg()
            return x.flip(-1)
        else:
            return x

    @staticmethod
    def transform_quantity(x: torch.Tensor, rot: int, flip: bool):
        if flip:
            return x.flip(-1).rot90(rot, [-2, -1])
        else:
            return x.rot90(rot, [-2, -1])


class BaseNorm(object):
    mean: Union[float, torch.Tensor]
    std: Union[float, torch.Tensor]

    def __call__(self, x: torch.Tensor):
        return (x - self.mean) / self.std

    def inverse(self, x: torch.Tensor):
        return x * self.std + self.mean

    def __repr__(self):
        return f'{self.__class__.__name__} object with mean = {self.mean} and std = {self.std}'


class PixelNorm(BaseNorm):
    def __init__(self, x: torch.Tensor, eps=1e-6, transforms=True, velocity=True):
        mean_ = x.mean(dim=0, keepdim=True)
        if velocity:
            self.mean = torch.mean(torch.cat([SeaIceTransform.transform_velocity(mean_, rot=r, flip=f) for f in (False, True) for r in range(4)]), dim=0, keepdim=True)
        else:
            self.mean = torch.mean(torch.cat([SeaIceTransform.transform_quantity(mean_, rot=r, flip=f) for f in (False, True) for r in range(4)]), dim=0, keepdim=True)

        x_ = (x - self.mean).square().mean(dim=0, keepdim=True)
        if velocity:
            self.std = torch.mean(torch.cat(
                [SeaIceTransform.transform_velocity(x_, rot=r, flip=f) for f in (False, True) for r in range(4)]).abs(),
                                  dim=0, keepdim=True).sqrt()
        else:
            self.std = torch.mean(torch.cat(
                [SeaIceTransform.transform_quantity(x_, rot=r, flip=f) for f in (False, True) for r in range(4)]).abs(),
                                  dim=0, keepdim=True).sqrt()
        self.std += eps


class ChannelNorm(BaseNorm):
    dims = (0, 2, 3)

    def __init__(self, x: torch.Tensor, eps=1e-6, transforms=True, velocity=True):
        mean_ = x.mean(dim=self.dims, keepdim=True)
        if velocity:
            self.mean = torch.mean(torch.cat(
                [SeaIceTransform.transform_velocity(mean_, rot=r, flip=f) for f in (False, True) for r in
                 range(4)]), dim=self.dims, keepdim=True)
        else:
            self.mean = torch.mean(torch.cat(
                [SeaIceTransform.transform_quantity(mean_, rot=r, flip=f) for f in (False, True) for r in
                 range(4)]), dim=self.dims, keepdim=True)

        x_ = (x - self.mean).square().mean(dim=self.dims, keepdim=True)
        if velocity:
            self.std = torch.mean(torch.cat(
                [SeaIceTransform.transform_velocity(x_, rot=r, flip=f) for f in (False, True) for r in
                 range(4)]).abs(), dim=self.dims, keepdim=True).sqrt()
        else:
            self.std = torch.mean(torch.cat(
                [SeaIceTransform.transform_quantity(x_, rot=r, flip=f) for f in (False, True) for r in
                 range(4)]).abs(), dim=self.dims, keepdim=True).sqrt()
        self.std += eps


class InstanceNorm(BaseNorm):
    dims = (2, 3)

    def __init__(self, x: torch.Tensor, eps=1e-6, transforms=True, velocity=True):
        self.dims = (2, 3)
        self.mean = x.mean(dim=self.dims, keepdim=True)
        self.std = torch.std(x, dim=self.dims, keepdim=True)
        self.std += eps


class MinMaxNorm(BaseNorm):
    dims = (0, 2, 3)

    def __init__(self, x: Optional[torch.Tensor] = None, eps=1e-6, transforms=True, velocity=True):
        self.mean = 0.
        if x is not None:
            self.mean = 0.
            self.std = x.abs().amax(dim=self.dims, keepdim=True) - self.mean
        else:
            self.mean = 0.
            self.std = 1.


class LabelScaling(BaseNorm):
    def __init__(self, norm: BaseNorm):
        self.norm = norm

    def __call__(self, x: torch.Tensor):
        return torch.tanh(self.norm(x) * 10)

    def inverse(self, x: torch.Tensor):
        return self.norm.inverse(torch.arctanh(x).clip(-10, 10) / 10)

    @property
    def mean(self):
        return self.norm.mean

    @mean.setter
    def mean(self, val):
        self.norm.mean = val

    @property
    def std(self):
        return self.norm.std

    @std.setter
    def std(self, val):
        self.norm.std = val

    def __repr__(self):
        return f'LabelScaling using {self.norm}'


class FourierData(SeaIceDataset):
    dt = 2.

    def __init__(self, basedir: str, transform: Optional[SeaIceTransform] = None, dev: torch.device = 'cpu',
                 phys_i: int = 0, scaling: Optional[dict[str, Tuple]] = None):
        self.transform = transform
        dirs, fn_c, self.names = zip(*[(dn, os.path.join(dn, "coef.param"), d) for s in os.listdir(basedir) if
                                       os.path.isdir(subdir := os.path.join(basedir, s)) for d in
                                       os.listdir(subdir) if os.path.isdir(dn := os.path.join(subdir, d))])

        raw_velocities = [torch.load(os.path.join(d, f'v{n}.tensor')) for d, n in zip(dirs, self.names)]

        self.t = [list(range(phys_i, len(v))) for v in raw_velocities]
        self.velocities = torch.cat([v[phys_i:] for v in raw_velocities]).to(dev)
        is_label, is_data = zip(*[(i != phys_i, i != max(t)) for t in self.t for i in t])

        self.data = self.velocities[list(is_data)]
        self.label = self.velocities[list(is_label)] - self.data

        raw_H = [torch.load(os.path.join(d, f'H{n}.tensor')) for d, n in zip(dirs, self.names)]
        self.H = torch.cat([h[phys_i:] for h in raw_H]).to(dev)
        raw_A = [torch.load(os.path.join(d, f'A{n}.tensor')) for d, n in zip(dirs, self.names)]
        self.A = torch.cat([a[phys_i:] for a in raw_A]).to(dev)

        self.v_a = torch.empty_like(self.data)

        x, y = torch.meshgrid(torch.linspace(0, 512, 257), torch.linspace(0, 512, 257), indexing='ij')
        x = x.reshape(1, 257, 257) / 1e3
        y = y.reshape(1, 257, 257) / 1e3

        j = 0
        for i, fn in enumerate(fn_c):
            coef = CoefficientDict()
            f = open(fn)
            while f.readline().rstrip() != '//Block Coefficients':
                pass
            while True:
                line = f.readline()
                if line == '\n' or line == '//Block Cyclone\n':
                    continue
                elif line == '' or line == '//Block Nix\n':
                    break

                words = line.rstrip().split()
                if len(words) == 2:  # Float
                    coef[words[0]] = float(words[1])
                elif len(words) > 2:  # List
                    coef[words[0]] = [float(x) for x in words[2:]]
            f.close()

            ind = slice(j, j + len(self.t[i]))
            self.v_a[ind] = self.cyclone(coef, x, y, [self.dt * k for k in self.t[i]])
            j += len(self.t[i])

        # Get transforms
        if scaling:
            self.data_scaling = MinMaxNorm()
            self.label_scaling = LabelScaling(MinMaxNorm())
            self.v_a_scaling = MinMaxNorm()
            self.H_scaling = MinMaxNorm()
            self.A_scaling = MinMaxNorm()
            self.data_scaling.mean = scaling['data'][0]
            self.data_scaling.std = scaling['data'][1]
            self.label_scaling.mean = scaling['label'][0]
            self.label_scaling.std = scaling['label'][1]
            self.v_a_scaling.mean = scaling['v_a'][0]
            self.v_a_scaling.std = scaling['v_a'][1]
            self.H_scaling.mean = scaling['H'][0]
            self.H_scaling.std = scaling['H'][1]
            self.A_scaling.mean = scaling['A'][0]
            self.A_scaling.std = scaling['A'][1]
        else:
            self.data_scaling = MinMaxNorm(self.data)
            self.label_scaling = LabelScaling(MinMaxNorm(self.label))
            self.v_a_scaling = MinMaxNorm(self.v_a)
            self.H_scaling = MinMaxNorm(self.H, velocity=False)
            self.A_scaling = MinMaxNorm(self.A, velocity=False)

        # Perform scaling
        self.data = self.data_scaling(self.data)
        self.label = self.label_scaling(self.label)
        self.v_a = self.v_a_scaling(self.v_a)
        self.H = self.H_scaling(self.H)
        self.A = self.A_scaling(self.A)
        print(f'data scaling: {self.data_scaling}\nlabel scaling: {self.label_scaling}\n'
              f'v_a scaling: {self.v_a_scaling}\nH scaling: {self.H_scaling}\nA scaling: {self.A_scaling}')

    def get_test_train_split(self, frac: float = .2, output: Optional[str] = None):
        split_idx = int(len(self.t) * frac)
        idx = torch.randperm(len(self.t))
        test_idx = idx[:split_idx]
        train_idx = idx[split_idx:]
        m = 0
        indices: list[list[int]] = [list(range(m, m := (m+len(tx)))) for tx in self.t]
        test_indices = [i for idx in test_idx for i in indices[idx]]
        train_indices = [i for idx in train_idx for i in indices[idx]]
        if output:
            with open(f"test_data_{output}.txt", "w") as f:
                f.write('\n'.join(np.sort([str(self.names[i]) for i in test_idx])))
        return torch.utils.data.Subset(self, test_indices), torch.utils.data.Subset(self, train_indices)

    @staticmethod
    def cyclone(coef: dict, x: torch.Tensor, y: torch.Tensor, t: list[float]):
        t = torch.tensor(t).view(-1, 1, 1)
        mx = coef['W_mx'] + coef['W_vx_m'] * t
        my = coef['W_my'] + coef['W_vy_m'] * t
        vmax = coef['W_vmax']
        alpha = coef['W_alpha']
        r0 = coef['W_r0']
        cyclone = coef['W_cyclone']
        ws = 1. if cyclone else -1.
        c = vmax * ws / r0 / exp(-1.)

        er = torch.exp(-torch.sqrt((x - mx).square() + (y - my).square()) / r0)
        res = torch.empty(len(t), 2, *x.shape[-2:])
        res[:, 0] = c * (cos(alpha) * (x - mx) + sin(alpha) * (y - my)) * er
        res[:, 1] = c * (-sin(alpha) * (x - mx) + cos(alpha) * (y - my)) * er

        return res

    @staticmethod
    def fourier_sum_xy(coef: dict, var: str, x: torch.Tensor, y: torch. Tensor):
        i_x_c = coef[var + '_i_x_c']
        c_x_c = coef[var + '_x_c']
        i_x_s = coef[var + '_i_x_s']
        c_x_s = coef[var + '_x_s']
        i_y_c = coef[var + '_i_y_c']
        c_y_c = coef[var + '_y_c']
        i_y_s = coef[var + '_i_y_s']
        c_y_s = coef[var + '_y_s']
        lb = coef[var + '_min']
        ub = coef[var + '_max']

        sigma = (sum([abs(c) for c in c_x_c]) + sum([abs(c) for c in c_x_s])) * (
                    sum([abs(c) for c in c_y_c]) + sum([abs(c) for c in c_y_s]))
        scaling = (ub - lb) / (2 * sigma)
        intercept = (ub - lb) / 2 + lb

        res_x = torch.stack([c_x_c[i] * torch.cos(i_x_c[i] * x) for i in range(len(i_x_c))]).sum(dim=0) if len(
            i_x_c) > 0 else torch.zeros_like(x)
        res_x += torch.stack([c_x_s[i] * torch.sin(i_x_s[i] * x) for i in range(len(i_x_s))]).sum(dim=0) if len(
            i_x_s) > 0 else torch.zeros_like(x)
        res_y = torch.stack([c_y_c[i] * torch.cos(i_y_c[i] * y) for i in range(len(i_y_c))]).sum(dim=0) if len(
            i_y_c) > 0 else torch.zeros_like(y)
        res_y += torch.stack([c_y_s[i] * torch.sin(i_y_s[i] * y) for i in range(len(i_y_s))]).sum(dim=0) if len(
            i_y_s) > 0 else torch.zeros_like(y)
        res = res_x * res_y
        res *= scaling
        res += intercept
        return res


class BenchData(SeaIceDataset):
    def __init__(self, basedir: str, steps: List[int], dev: torch.device = 'cpu'):
        self.velocities = read_velocities([f"{basedir}v.{n:05}.vtk" for n in steps]).to(dev)
        self.data = SeaIceDataset.scale_velocity(self.velocities[:-1])
        self.label = SeaIceDataset.scale_velocity(self.velocities[1:]) - self.data

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
        self.v_a = -self.v_a * 1e-1  # scaling with *1e-3(dimensionless) *1e4(velocity correction) *1e-2 (typical size)

        self.v_o = torch.empty_like(self.data)
        self.v_o[:, 0] = .01 * (y / 250 - 1) * 1e2  # scaling with *1e-3(dimensionless) *1e4(velocity correction) *1e1 (typical size)
        self.v_o[:, 1] = .01 * (1 - x / 250) * 1e2


class PatchData(BenchData):
    def __init__(self, basedir: str, steps: List[int], patch_size: int = 3, overlap: int = 1,
                 dev: torch.device = 'cpu'):
        super().__init__(basedir, steps, dev)
        self.data_t, self.H_t, self.A_t, self.v_a_t, self.v_o_t, self.border_chunks, self.label_t = transform_data(
            self.data, self.H, self.A, self.v_a, self.v_o, self.label, patch_size, overlap)

    def __getitem__(self, index) -> T_co:
        return (self.data_t[index], self.H_t[index], self.A_t[index], self.v_a_t[index],
                self.v_o_t[index], self.border_chunks[index], self.label_t[index])

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


def get_patches(im: torch.Tensor, patch_size: int = 3, overlap: int = 1, vertex: bool = True):
    output_patch_size = patch_size + 2 * overlap if vertex else patch_size - 1 + 2 * overlap
    patches = im.unfold(2, output_patch_size, patch_size)
    patches = patches.unfold(3, output_patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, im.shape[1], output_patch_size, output_patch_size)
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


def transform_data(data, H, A, v_a, v_o, label, patch_size: int = 3, overlap: int = 1):
    # Add a first dimension in case the data is not batched
    if data.dim() == 3:
        data = data[None]
        H = H[None]
        A = A[None]
        v_a = v_a[None]
        v_o = v_o[None]
        label = label[None]

    # If overlap is larger than 1, pad the input extra
    if overlap > 1:
        input_padding = int(overlap - 1)
        input_padding = (input_padding, input_padding, input_padding, input_padding)
        data = torch.nn.functional.pad(data, input_padding)
        H = torch.nn.functional.pad(H, input_padding)
        A = torch.nn.functional.pad(A, input_padding)
        v_a = torch.nn.functional.pad(v_a, input_padding)
        v_o = torch.nn.functional.pad(v_o, input_padding)

    # Check if data has right format and compute number of patches to be created
    assert overlap > 0, "overlap needs to be greater than 0"
    assert (data.shape[2] - 2 * overlap) % patch_size == 0, "Height cannot be divided using chunk_size and overlap"
    assert (data.shape[3] - 2 * overlap) % patch_size == 0, "Width cannot be divided using chunk_size and overlap"
    n_chunks = int((data.shape[2] - 2 * overlap) / patch_size)

    # Get patches
    data = get_patches(data, patch_size, overlap)
    v_a = get_patches(v_a, patch_size, overlap)
    v_o = get_patches(v_o, patch_size, overlap)
    label = get_patches(label[:, :, 1:-1, 1:-1], patch_size, 0)

    H = get_patches(H, patch_size, overlap, vertex=False)
    A = get_patches(A, patch_size, overlap, vertex=False)

    # Create boolean tensor describing if the patch is a border patch
    border_chunks = torch.arange(data.shape[0], device=data.device).reshape(-1, n_chunks, n_chunks)
    border_chunks = (border_chunks % n_chunks == 0) + (border_chunks % n_chunks == (n_chunks - 1))
    border_chunks = border_chunks + border_chunks.transpose(1, 2)
    border_chunks = border_chunks.reshape(-1, 1, 1, 1).to(torch.float32)
    return data, H, A, v_a, v_o, border_chunks, label


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Specifies the path used for the data', type=str)
    parser.add_argument('storage', help='Specifies the path used for the data', type=str)
    args = parser.parse_args()
    data_path = args.path
    storage_path = args.storage

    dirs = [(dn, d) for s in os.listdir(data_path) if
            os.path.isdir(subdir := os.path.join(data_path, s)) for d in
            os.listdir(subdir) if os.path.isdir(dn := os.path.join(subdir, d))]
    for d, n in tqdm(dirs):
        if os.path.isfile(os.path.join(storage_path, n, f'v{n}.tensor')):
            continue
        fn_v = [os.path.join(d, fn) for fn in list(filter(lambda f: f[0] == 'v' and f[-3:] == 'vtk', sorted(os.listdir(d))))]
        velocities = read_velocities(fn_v, True)
        torch.save(velocities, os.path.join(storage_path, n, f'v{n}.tensor'))

        fn_dgh = [os.path.join(d, fn) for fn in list(filter(lambda f: f[:3] == 'dgh' and f[-3:] == 'vtk', sorted(os.listdir(d))))]
        H, A = read_HA(fn_dgh, True)
        torch.save(H, os.path.join(storage_path, n, f'H{n}.tensor'))
        torch.save(A, os.path.join(storage_path, n, f'A{n}.tensor'))

