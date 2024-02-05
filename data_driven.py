from math import ceil
import torch
from math import ceil

import torch
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import FourierData, SeaIceTransform
from loss import Loss
from surrogate_net import SurrogateNet, UNet


# from visualisation import plot_comparison, plot_losses
# from torchvision.utils import make_grid
# from dataset import transform_data


def train(model, dataset, dev, n_steps=128, main_loss='MSE', job_id=None):
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [.9, .1])
    dataloader = DataLoader(train_dataset, batch_size=max(8, ceil(len(train_dataset) / 10)), shuffle=True)

    criterion = Loss(main_loss).to(dev)
    test_criterion = Loss(main_loss).to(dev)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[120000, 170000], gamma=.1)

    model_id = f"{model.__class__.__name__}_{main_loss}"
    if job_id:
        model_id += f"_{job_id}"
    else:
        from datetime import datetime
        stamp = datetime.now().strftime('%m%d%H%M%S')
        model_id += f"_{stamp}".replace(' ', '')

    pbar = trange(n_steps, mininterval=60.)
    for i in pbar:
        for (data, H, A, v_a, v_o, label) in dataloader:
            # Forward pass and compute output
            output = model(data, H, A, v_a, v_o)

            # Compute losses
            loss = criterion(output, label, data, A, store=True)

            # Gradient computation and optimiser step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optim.step()
            torch.cuda.empty_cache()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            test_output = model(*test_dataset[:][:-1])
            test_loss = test_criterion(test_output, test_dataset[:][-1], test_dataset[:][0], test_dataset[:][2],
                                       store=True, grad=False)
        model.train()

        if i % 5 == 0:
            torch.save(model, f'model_{model_id}.pt')
            torch.save(criterion.results, f'losses_{model_id}.li')
            torch.save(test_criterion.results, f'test_losses_{model_id}.li')

        pbar.set_postfix(test_loss=test_loss.item(), refresh=False)

    torch.save(model, f'model_{model_id}.pt')
    torch.save(criterion.results, f'losses_{model_id}.li')
    torch.save(test_criterion.results, f'test_losses_{model_id}.li')

    return model, criterion.results


if __name__ == "__main__":
    import sys

    job_id = None
    data_path = (sys.argv[1])
    n_steps = int(sys.argv[2])
    main_loss = str(sys.argv[3])
    model_name = str(sys.argv[4])  # 'UNet' or else SurrogateNet
    if len(sys.argv) == 6:
        job_id = sys.argv[5]

    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = UNet().to(dev) if model_name == 'UNet' else SurrogateNet().to(dev)

    dataset = FourierData(data_path, SeaIceTransform(), dev=dev)

    model, results = train(model, dataset, dev, n_steps, main_loss, job_id)

    # # Plot results
    # plot_comparison(model, dataset, 0, 0, patch_size, overlap)
    # plot_losses(results['loss'], results['mean'], results['std'], results['contrast'], results['classic'],
    #             names=['loss', 'mean', 'std', 'contrast', 'classic'])
