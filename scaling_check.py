import torch

from dataset import FourierData, SeaIceTransform

if __name__ == "__main__":
    import argparse
    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Specifies the path used to the data', type=str)

    parser.add_argument('-p', '--physical', type=int, default=10, help='Time step after which physical states are assumed')

    args = parser.parse_args()

    data_path = args.path
    phys_i = args.physical
    dataset = FourierData(data_path, SeaIceTransform(), dev=dev, phys_i=phys_i)
