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
    parser.add_argument('--method', choices=['agnostic', 'clientwise', 'ours'])     # method name
    parser.add_argument('--home')                                                   # home folder location
    parser.add_argument('--numSeeds', default=10, type=int)                         # number of different seeds
    parser.add_argument('--numComRnds', default=100, type=int)                      # number of communcation rounds
    parser.add_argument('--numLambdas', default=50, type=int)                       # number of lambdas to check
    parser.add_argument('--runName', default='run')                                 # run name for weights and biases

    args = parser.parse_args()
    HOMEFOLDER = args.home
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using Device')
    print(DEVICE)

    method = args.method  # agnostic, clientwise, ours
    ls = np.logspace(-5, 2, args.numLambdas)
    for mu in [1.0]:
        for NY in [20]:
            for l in ls:
                for seed in range(args.numSeeds):
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    if method == 'agnostic':
                        s = agnostic_server.Server(datasets.CommunitiesCrimeDataset().load_data(HOMEFOLDER=HOMEFOLDER), buildModelClass(99), torch.nn.BCEWithLogitsLoss(), m=None, T=args.numComRnds, lambda_=l, datasetname='CommunitiesCrime', runname=args.runName, device=DEVICE)
                    elif method == 'clientwise':
                        s = clientwise_server.Server(datasets.CommunitiesCrimeDataset().load_data(HOMEFOLDER=HOMEFOLDER), buildModelClass(99), torch.nn.BCEWithLogitsLoss(), m=None, T=args.numComRnds, lambda_=l, datasetname='CommunitiesCrime', runname=args.runName, device=DEVICE)
                    elif method == 'ours':
                        s = our_server.Server(datasets.CommunitiesCrimeDataset().load_data(HOMEFOLDER=HOMEFOLDER), buildModelClass(99), torch.nn.BCEWithLogitsLoss(), m=None, T=args.numComRnds, mu=mu, NY=NY, lambda_=l, datasetname='CommunitiesCrime', runname=args.runName, device=DEVICE)
                    else:
                        raise NotImplementedError(f'Method {method} currently not supported')
                    s.train_test_split()
                    s.sync_N()
                    if method == 'ours':
                        s.sync_Pa()
                    s.train()
                    res, weights = s.test_current_model()
                    performances = np.vstack([np.array([metrics.accuracy(p, y).cpu(), metrics.P1(p, a).cpu()]) for p, y, a in res])
                    np.save(os.path.join(HOMEFOLDER, f'results/{method}/cc_p_{l}_{seed}_{NY}.npy'), performances)
