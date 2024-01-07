import os
from math import ceil
import torch
from torch.utils.data import DataLoader
# from tqdm import trange

from dataset import FourierData
from loss import strain_rate_loss
from surrogate_net import SurrogateNet
# from visualisation import plot_comparison, plot_losses
# from torchvision.utils import make_grid
# from dataset import transform_data


def train(model, dataset, dev, n_steps=128, strain_weight=1, job_id=None):
    dataloader = DataLoader(dataset, batch_size=max(8, ceil(len(dataset) / 10)), shuffle=True)

    criterion = torch.nn.MSELoss().to(dev)
    structure_criterion = torch.nn.MSELoss().to(dev)
    inst_norm = torch.nn.InstanceNorm2d(2).to(dev)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[2000, 10000], gamma=.1)
    losses = []
    mean_losses = []
    std_losses = []
    contrast_losses = []
    classic_losses = []
    strain_losses = []

    pbar = range(n_steps)
    for i in pbar:
        for (data, H, A, v_a, v_o, label) in dataloader:
            # Forward pass and compute output
            output = model(data, H, A, v_a, v_o)
            contrast = inst_norm(output)
            m = output.mean(dim=(2, 3), keepdim=True)
            s = output.std(dim=(2, 3), keepdim=True, unbiased=False)

            # Compute losses
            contrast_loss = structure_criterion(contrast, inst_norm(label))
            classic_loss = criterion(output, label)
            strain_loss = strain_rate_loss(output, label)
            loss = classic_loss + strain_weight * strain_loss
            with torch.no_grad():
                mean_loss = (m - label.mean(dim=(2, 3), keepdim=True)).square().mean()
                std_loss = (s - label.std(dim=(2, 3), keepdim=True, unbiased=False)).square().mean()

            # Store losses
            losses.append(loss.item())
            mean_losses.append(mean_loss.item())
            std_losses.append(std_loss.item())
            contrast_losses.append(contrast_loss.item())
            classic_losses.append(classic_loss.item())
            strain_losses.append(strain_loss.item())

            # Gradient computation and optimiser step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optim.step()
            torch.cuda.empty_cache()
            scheduler.step()
        # pbar.set_postfix(loss=losses[-1], contrast_loss=contrast_losses[-1])  # , PINN=PINN_losses[-1])

    from datetime import datetime
    stamp = datetime.now().strftime('%m%d%H%M%S')
    model_id = f"{n_steps}_{stamp}".replace(' ', '')
    if job_id:
        model_id += f"_{job_id}"
    torch.save(model.cpu(), f'model_{model_id}.pt')
    results = {'loss': losses, 'mean': mean_losses, 'std': std_losses, 'contrast': contrast_losses,
               'classic': classic_losses, 'strain': strain_losses}
    torch.save(results, f'losses_{model_id}.li')

    return model, results


if __name__ == "__main__":
    import sys
    job_id = None
    complexity = 0
    strain_weight = 1.
    if len(sys.argv) >= 9:
        data_path = (sys.argv[1])
        # patch_size = int(sys.argv[2])
        # overlap = int(sys.argv[3])
        hidden = (int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
        complexity = int(sys.argv[5])
        save_dataset = bool(int(sys.argv[6]))
        n_steps = int(sys.argv[7])
        strain_weight = float(sys.argv[8])
        if len(sys.argv) == 10:
            job_id = sys.argv[9]
    else:
        data_path = 'C:\\Users\\Brend\\PycharmProjects\\MasterThesis\\data\\data\\'
        hidden = (1, 2, 3)
        save_dataset = False
        n_steps = 128

    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = SurrogateNet().to(dev)

    if os.path.isfile(f'full_dataset.data'):
        dataset = torch.load(f'full_dataset.data')
    else:
        dataset = FourierData(data_path, dev=dev)
        if save_dataset:
            torch.save(dataset, f'full_dataset.data')

    model, results = train(model, dataset, dev, n_steps, strain_weight, job_id)

    # # Plot results
    # plot_comparison(model, dataset, 0, 0, patch_size, overlap)
    # plot_losses(results['loss'], results['mean'], results['std'], results['contrast'], results['classic'],
    #             names=['loss', 'mean', 'std', 'contrast', 'classic'])
