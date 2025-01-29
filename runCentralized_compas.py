import os
import torch
import numpy

from src.utils import datasets
from src.centralized import train

from src.utils import metrics
import numpy as np
import argparse


def buildModelClass(inputSize):
    class NNModel(torch.nn.Module):
        def __init__(self, inputSize=inputSize, outputSize=1):
            super(NNModel, self).__init__()
            self.linear1 = torch.nn.Linear(inputSize, 16, bias=True)
            self.linear2 = torch.nn.Linear(16, outputSize, bias=True)

        def forward(self, x):
            x = torch.nn.functional.relu(self.linear1(x))
            out = self.linear2(x)
            return out
    return NNModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--home')                                                   # home folder location
    parser.add_argument('--numSeeds', default=10, type=int)                         # number of different seeds
    parser.add_argument('--numEpochs', default=1000, type=int)                       # number of epochs rounds
    parser.add_argument('--numLambdas', default=50, type=int)                       # number of lambdas to check
    parser.add_argument('--runName', default='Centralized')                           # run name for weights and biases

    args = parser.parse_args()
    HOMEFOLDER = args.home
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ls = np.logspace(-5, 1, args.numLambdas)
    for l in ls:
        for seed in range(args.numSeeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            trainer = train.Trainer(datasets.CompasDataset().load_data(HOMEFOLDER=HOMEFOLDER), buildModelClass(8), torch.nn.BCEWithLogitsLoss(), N_epochs=args.numEpochs, lambda_=l, datasetname='Compas')
            trainer.train()
