import argparse
import os
from glob import glob

import torch
from torch.utils.data import DataLoader

from bayesian_train import save_result
from dataset import FourierData, SeaIceTransform
from loss import Loss, shear_l1_loss

import numpy as np


if __name__ == '__main__':
    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Specifies the path used to the data', type=str)
    parser.add_argument('dir', help='Specifies the directory with models', type=str, default='.')
    parser.add_argument('loss', help='Name of the loss that is saved to register', type=str)
    parser.add_argument('weight', type=float, default=1e-4, help='Weight of SRE/MRE loss')
    parser.add_argument('epsilon', type=float, default=1e-4, help='MRE epsilon')
    args = parser.parse_args()

    data_path = args.path
    directory = args.dir
    loss_name = args.loss
    weight = 10 ** args.weight
    eps = 10 ** args.epsilon

    ids = np.atleast_2d(np.loadtxt(os.path.join(directory, 'register.txt')))[:, 0]

    dataset = FourierData(data_path, SeaIceTransform(), dev=dev, phys_i=10, max_size=None)
    loss = Loss(dataset.scalings, loss_name, mre_eps=eps, weight=weight).to(dev)

    for id in ids:
        model = torch.load(glob(os.path.join(directory, 'model*' + id + '.pt'))[0]).to(dev).eval()
        with torch.no_grad():
            with open(glob(os.path.join(directory, f'test_data_{id}.txt'))[0], 'r', encoding='latin-1') as file:
                test_names = file.read().split()
            test_split = [np.where(n == np.array(dataset.names))[0][0] for n in
                          np.intersect1d(dataset.names, test_names)]
            data_loader = DataLoader(dataset[test_split], batch_size=64, shuffle=False)
            result = []
            for t_data, t_H, t_A, t_v_a, t_v_o, t_label in data_loader:
                out = model(t_data, t_H, t_A, t_v_a, t_v_o)
                result.append(shear_l1_loss(dataset.label_scaling.inverse(out) + dataset.data_scaling.inverse(t_data),
                                            dataset.label_scaling.inverse(t_label) + dataset.data_scaling.inverse(t_data),
                                            reduction='none'))

        # Aggregate losses
        value = torch.cat(result).mean(dim=(1, 2, 3))
        # Save to register_new.txt
        save_result(id, value, 'register_new.txt')

