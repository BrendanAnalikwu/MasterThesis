from math import ceil
from typing import Optional, Tuple

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


def train(model, dataset, dev, n_steps=128, main_loss='MSE', job_id=None, betas=(.9, .999), batch_size=8):
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [.8, .2])
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    criterion = Loss(main_loss).to(dev)
    test_criterion = Loss(main_loss).to(dev)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2, betas=betas)
    scheduler = ReduceLROnPlateau(optim, patience=200, min_lr=1e-9)
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

    pbar = trange(n_steps, mininterval=1.)
    for i in pbar:
        for (data, H, A, v_a, label) in dataloader:
            # Forward pass and compute output
            output = model(data, H, A, v_a)

            # Compute losses
            reg = 0
            for w in model.parameters():
                reg += w.norm(1)
            regs.append((reg / reg_n).item())
            loss = criterion(output, label, data, A, store=True) + .1 * reg / reg_n

            # Gradient computation and optimiser step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optim.step()
            torch.cuda.empty_cache()

            scheduler.step(loss.item())
            if optim.param_groups[0]['lr'] != last_lr:
                last_lr = optim.param_groups[0]['lr']
                print(f"NEW LEARNING RATE: {last_lr}")
                pbar.update()

        model.eval()
        with torch.no_grad():
            t_data, t_H, t_A, t_v_a, t_label = test_dataset[:]
            test_output = model(t_data, t_H, t_A, t_v_a)
            test_criterion(test_output, t_label, t_data, t_A, store=True, grad=False)
        model.train()

        if i % 5 == 0:
            torch.save(model, f'model_{model_id}.pt')
            torch.save(criterion.results, f'losses_{model_id}.li')
            torch.save(test_criterion.results, f'test_losses_{model_id}.li')

        pbar.set_postfix(loss=criterion.results[main_loss][-1], test_loss=test_criterion.results[main_loss][-1], lr=last_lr)

    torch.save(model, f'model_{model_id}.pt')
    torch.save(criterion.results, f'losses_{model_id}.li')
    torch.save(test_criterion.results, f'test_losses_{model_id}.li')

    return model, criterion.results


if __name__ == "__main__":
    import sys

    data_path = (sys.argv[1])
    n_steps = int(sys.argv[2])
    main_loss = str(sys.argv[3])
    model_name = str(sys.argv[4])  # 'UNet' or else SurrogateNet
    job_id = sys.argv[5]
    betas = float(sys.argv[6]), float(sys.argv[7])
    batch_size = 16
    if len(sys.argv) == 9:
        batch_size = int(sys.argv[8])

    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if model_name == 'UNet':
        model = UNet().to(dev)
    elif model_name == 'Noisy':
        model = NoisySurrogateNet().to(dev)
    elif model_name == 'Small':
        model = SmallSurrogateNet().to(dev)
    else:
        model = SurrogateNet().to(dev)

    dataset = FourierData(data_path, SeaIceTransform(), dev=dev, phys_i=10)

    model, results = train(model, dataset, dev, n_steps, main_loss, job_id, betas, batch_size)

    # # Plot results
    # plot_comparison(model, dataset, 0, 0, patch_size, overlap)
    # plot_losses(results['loss'], results['mean'], results['std'], results['contrast'], results['classic'],
    #             names=['loss', 'mean', 'std', 'contrast', 'classic'])
