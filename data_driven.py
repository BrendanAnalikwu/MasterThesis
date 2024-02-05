import os
from math import ceil
import torch
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import FourierData, SeaIceTransform
from loss import strain_rate_loss, mean_concentration_loss, mean_relative_loss
from surrogate_net import SurrogateNet, UNet


# from visualisation import plot_comparison, plot_losses
# from torchvision.utils import make_grid
# from dataset import transform_data


def train(model, dataset, dev, n_steps=128, strain_weight=1., job_id=None):
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [.9, .1])
    dataloader = DataLoader(train_dataset, batch_size=max(8, ceil(len(train_dataset) / 10)), shuffle=True)

    criterion = torch.nn.MSELoss().to(dev)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[120000, 170000], gamma=.1)
    losses = []
    classic_losses = []
    strain_losses = []
    test_losses = []
    MCE_losses = []
    MRE_losses = []
    MRE_full_losses = []

    from datetime import datetime
    stamp = datetime.now().strftime('%m%d%H%M%S')
    model_id = f"{n_steps}_{stamp}".replace(' ', '')
    if job_id:
        model_id += f"_{job_id}"

    pbar = trange(n_steps, mininterval=60.)
    for i in pbar:
        for (data, H, A, v_a, v_o, label) in dataloader:
            # Forward pass and compute output
            output = model(data, H, A, v_a, v_o)

            # Compute losses
            classic_loss = criterion(output, label)
            strain_loss = strain_rate_loss(output, label)
            loss = classic_loss + strain_weight * strain_loss
            with torch.no_grad():
                MCE_loss = mean_concentration_loss(output, label, data, A)
                MRE_loss = mean_relative_loss(output, label)
                MRE_full_loss = mean_relative_loss(output, label, data)

            # Store losses
            losses.append(loss.item())
            classic_losses.append(classic_loss.item())
            strain_losses.append(strain_loss.item())
            MCE_losses.append(MCE_loss.item())
            MRE_losses.append(MRE_loss.item())
            MRE_full_losses.append(MRE_full_loss.item())

            # Gradient computation and optimiser step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optim.step()
            torch.cuda.empty_cache()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            test_output = model(*test_dataset[:][:-1])
            test_losses.append(criterion(test_output, test_dataset[:][-1]).item())
        model.train()

        if i % 5 == 0:
            torch.save(model, f'model_{model_id}.pt')
            results = {'loss': losses, 'classic': classic_losses, 'strain': strain_losses, 'test': test_losses,
                       'concentration': MCE_losses, 'relative': MRE_losses, 'relative_full': MRE_full_losses}
            torch.save(results, f'losses_{model_id}.li')

        pbar.set_postfix(test_loss=test_losses[-1], refresh=False)

    torch.save(model, f'model_{model_id}.pt')
    results = {'loss': losses, 'classic': classic_losses, 'strain': strain_losses,
               'MCE': MCE_losses, 'relative': MRE_losses, 'relative_full': MRE_full_losses}
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

    model = UNet().to(dev)

    dataset = FourierData(data_path, SeaIceTransform(), dev=dev)

    model, results = train(model, dataset, dev, n_steps, strain_weight, job_id)

    # # Plot results
    # plot_comparison(model, dataset, 0, 0, patch_size, overlap)
    # plot_losses(results['loss'], results['mean'], results['std'], results['contrast'], results['classic'],
    #             names=['loss', 'mean', 'std', 'contrast', 'classic'])
