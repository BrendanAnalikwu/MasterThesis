import argparse
import os
import warnings
from glob import glob

import numpy as np
import torch
from torch.utils.data import DataLoader

from bayesian_train import save_result
from dataset import FourierData
from loss import shear_l1_loss, mean_relative_loss, strain_rate_loss

if __name__ == '__main__':
    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Specifies the path used to the data', type=str)
    parser.add_argument('dir', help='Specifies the directory with models', type=str, default='.')
    parser.add_argument('loss', help='Name of the loss that is saved to register', type=str)
    args = parser.parse_args()

    data_path = args.path
    directory = args.dir
    loss_name = args.loss

    ids = np.atleast_2d(np.loadtxt(os.path.join(directory, 'register.txt')))[:, 0]

    dataset = FourierData(data_path, None, dev=dev, phys_i=10, max_size=None)
    with open(glob(os.path.join(directory, f'test_data_*.txt'))[0], 'r', encoding='latin-1') as file:
        test_names = file.read().split()

    for id in ids:
        try:
            model = torch.load(glob(os.path.join(directory, f'model*{int(id)}.pt'))[0]).to(dev).eval()
        except (EOFError, IndexError):
            warnings.warn(f"Model {int(id)} could not be loaded")
            continue
        with torch.no_grad():
            test_split = [np.where(n == np.array(dataset.names))[0][0] for n in
                          np.intersect1d(dataset.names, test_names)]
            data_loader = DataLoader(dataset.get_subset(test_split), batch_size=64, shuffle=False)
            result = []
            result_sre = []
            for t_data, t_H, t_A, t_v_a, t_v_o, t_label in data_loader:
                out = model(t_data, t_H, t_A, t_v_a, t_v_o)
                result.append(mean_relative_loss(out, t_label, t_data, eps=2.3107e-3, reduction='none'))
                result_sre.append(strain_rate_loss(out, t_label, reduction='none'))

        # Aggregate losses
        value = torch.cat(result).mean()
        # Save to register_new.txt
        save_result(id, value.item(), 'register_new.txt')

        value = torch.cat(result_sre).mean()
        save_result(id, value.item(), 'register_sre.txt')

