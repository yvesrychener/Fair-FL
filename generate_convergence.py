import os
import torch
import numpy

from src.utils import datasets
from src.agnostic import server as agnostic_server
from src.clientwise import server as clientwise_server
from src.ours import server as our_server

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
    parser.add_argument('--home')
    args = parser.parse_args()
    HOMEFOLDER = args.home
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using Device')
    print(DEVICE)
    lambda_ = 0.5
    NY = 100
    for seed in range(10):
        torch.manual_seed(seed)
        np.random.seed(seed)
        s = our_server.Server(datasets.CommunitiesCrimeDataset().load_data(HOMEFOLDER=HOMEFOLDER), buildModelClass(99), torch.nn.BCEWithLogitsLoss(), m=None, T=100, mu=1, NY=NY, lambda_=lambda_, datasetname='CommunitiesCrime', device=DEVICE, convergence=True)
        s.train_test_split()
        s.sync_N()
        s.sync_Pa()
        s.train()
        res, weights = s.test_current_model()
