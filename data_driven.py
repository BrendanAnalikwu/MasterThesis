from math import ceil
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import FourierData, SeaIceTransform
from loss import Loss
from surrogate_net import SurrogateNet, UNet, NoisySurrogateNet, SmallSurrogateNet


# from visualisation import plot_comparison, plot_losses
# from torchvision.utils import make_grid
# from dataset import transform_data
torch.manual_seed(0)
getCoefficientNorm = lambda layer: np.sqrt(1 / (layer.weight.shape[1] * layer.weight.shape[2] * layer.weight.shape[3]))


def getParameters(model: torch.nn.Module, lr: float = .1):
    params = []
    if len(list(model.children())) > 0:
        for layer in model.children():
            params.extend(getParameters(layer, lr))
    else:
        paramdict = {'params': model.parameters()}
        if isinstance(model, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            paramdict['lr'] = getCoefficientNorm(model) * lr
        params.append(paramdict)
    return params


def train(model, dataset, dev, n_steps=128, main_loss='MSE', job_id=None, betas=(.9, .999), batch_size=8, save=True):
    test_dataset, train_dataset = dataset.get_test_train_split(.2, job_id)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=len(train_dataset) > batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=len(test_dataset) > 64)

    criterion = Loss(main_loss).to(dev)
    test_criterion = Loss(main_loss).to(dev)
    optim = torch.optim.Adam(getParameters(model, 1e-3), lr=1e-3, betas=betas)
    scheduler = ReduceLROnPlateau(optim, patience=10, min_lr=1e-9)
    last_lr = optim.param_groups[0]['lr']

    model_id = f"{model.__class__.__name__}_{main_loss}"
    if job_id:
        model_id += f"_{job_id}"
    else:
        from datetime import datetime
        stamp = datetime.now().strftime('%m%d%H%M%S')
        model_id += f"_{stamp}".replace(' ', '')

    reg_n = sum(p.numel() for p in model.parameters())
    regs = []
    # params = [[p.abs().mean().cpu().detach() for p in model.parameters() if p.dim() > 3]]

    pbar = trange(n_steps)
    for i in pbar:
        for (data, H, A, v_a, label) in dataloader:
            # Forward pass and compute output
            # autoencoder_output = model.encoder(label, H, A, v_a)
            # encoding_output = model.encoder(data, H, A, v_a)
            output = model(data, H, A, v_a)

            # Compute losses
            with torch.no_grad():
                reg = 0
                for w in model.parameters():
                    reg += w.norm(1)
                regs.append((reg / reg_n).item())
            loss = criterion(output, label, data, A, store=True)  # + .1 * reg / reg_n

            # Gradient computation and optimiser step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optim.step()
            torch.cuda.empty_cache()

        scheduler.step(np.mean(criterion.results[main_loss][-len(dataloader):]))
        if optim.param_groups[0]['lr'] != last_lr:
            last_lr = optim.param_groups[0]['lr']
            print(f"NEW LEARNING RATE: {last_lr}")
            pbar.update()

        # params.append([p.abs().mean().cpu().detach() for p in model.parameters() if p.dim() > 3])
        model.eval()
        with torch.no_grad():
            for t_data, t_H, t_A, t_v_a, t_label in test_dataloader:
                test_output = model(t_data, t_H, t_A, t_v_a)
                test_criterion(test_output, t_label, t_data, t_A, store=False, grad=False)
            test_criterion.flush_stack_to_results()
        model.train()

        if i % 5 == 0 and save:
            torch.save(model, f'model_{model_id}.pt')
            torch.save(criterion.results, f'losses_{model_id}.li')
            torch.save(test_criterion.results, f'test_losses_{model_id}.li')

        pbar.set_postfix(loss=np.mean(criterion.results[main_loss][-len(dataloader):]),
                         test_loss=test_criterion.results[main_loss][-1],
                         mse=np.mean(criterion.results['MSE'][-len(dataloader):]),
                         lr=last_lr)

    torch.save(model, f'model_{model_id}.pt')
    torch.save(criterion.results, f'losses_{model_id}.li')
    torch.save(test_criterion.results, f'test_losses_{model_id}.li')

    return model, criterion.results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Specifies the path used to the data', type=str)
    parser.add_argument('n_steps', help='Number of epochs', type=int)
    parser.add_argument('main_loss', help='Name of the loss that is trained on', type=str, choices=Loss.loss_names)
    parser.add_argument('model_name', help='Name of the model', type=str, choices=['SurrogateNet', 'UNet', 'Noisy', 'Small'])
    parser.add_argument('-j', '--job_id', help='Slurm job id', type=str, default='0')
    parser.add_argument('-b', '--betas', type=float, nargs=2, default=[.9, .999])
    parser.add_argument('-B', '--batch_size', help='Batch size of mini-batches', type=int, default=16)
    parser.add_argument('-p', '--physical', type=int, default=10, help='Time step after which physical states are assumed')
    parser.add_argument('-s', '--nosave', help='Specifies if intermediate saving of losses and model is enables', action='store_false')

    args = parser.parse_args()

    data_path = args.path
    n_steps = args.n_steps
    main_loss = args.main_loss
    model_name = args.model_name  # 'UNet' or else SurrogateNet
    job_id = args.job_id
    betas = args.betas
    batch_size = args.batch_size
    phys_i = args.physical
    save = args.nosave

    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if model_name == 'UNet':
        model = UNet().to(dev)
    elif model_name == 'Noisy':
        model = NoisySurrogateNet().to(dev)
    elif model_name == 'Small':
        model = SmallSurrogateNet().to(dev)
    else:
        model = SurrogateNet().to(dev)

    dataset = FourierData(data_path, SeaIceTransform(), dev=dev, phys_i=phys_i)

    model, results = train(model, dataset, dev, n_steps, main_loss, job_id, betas, batch_size, save)

    # # Plot results
    # plot_comparison(model, dataset, 0, 0, patch_size, overlap)
    # plot_losses(results['loss'], results['mean'], results['std'], results['contrast'], results['classic'],
    #             names=['loss', 'mean', 'std', 'contrast', 'classic'])
