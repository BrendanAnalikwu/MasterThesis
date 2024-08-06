import sys
import warnings
from typing import Union

from data_driven import *


def save_result(job_id: int, value: Union[float, str, int], filename: str = 'register.txt'):
    with open(filename, 'ab') as f:
        np.savetxt(f, np.array([float(job_id), value])[None], fmt="%d %e")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Specifies the path used to the data', type=str)
    parser.add_argument('job_id', help='Slurm job id', type=str, default='0')
    parser.add_argument('model_name', help='Name of the model', type=int)
    parser.add_argument('main_loss', help='Name of the loss that is trained on', type=int)
    parser.add_argument('betas', type=float, nargs=2, default=[.9, .999])
    parser.add_argument('batch_size', help='Batch size of mini-batches', type=float, default=16)
    parser.add_argument('alpha', help='Regularisation alpha', type=float, default=1.)
    parser.add_argument('noiselvl', type=float, default=0., help='Amount of noise to add to training data')
    parser.add_argument('learning_rate', type=float, default=1e-4, help='Optimiser\'s learning rate')
    parser.add_argument('weight', type=float, default=1e-4, help='Weight of SRE/MRE loss')
    parser.add_argument('epsilon', type=float, default=1e-4, help='MRE epsilon')

    args = parser.parse_args()

    data_path = args.path
    main_loss = Loss.loss_names[args.main_loss]
    model_name = args.model_name
    job_id = args.job_id
    betas = (1 - 10 ** args.betas[0], 1 - 10 ** args.betas[1])
    batch_size = int(np.round(2 ** args.batch_size))
    alpha = 10 ** args.alpha
    noise_lvl = 10 ** args.noiselvl
    learning_rate = 10 ** args.learning_rate
    weight = 10 ** args.weight
    eps = 10 ** args.epsilon

    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if model_name == 0:
        model = UNet().to(dev)
    elif model_name == 1:
        model = SurrogateNet().to(dev)
    else:
        warnings.warn("Model specifier not 0 or 1. Defaulted to UNet")
        model = UNet().to(dev)

    dataset = FourierData(data_path, SeaIceTransform(), dev=dev, phys_i=10, max_size=None)

    model, results, test_results = train(model, dataset, dev, 10000, main_loss, job_id, betas, batch_size, alpha,
                                         noise_lvl, True, learning_rate, weight, eps)

    save_result(job_id, min(test_results['SL']))
    sys.exit(0)
