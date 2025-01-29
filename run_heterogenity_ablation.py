import os
import torch
import numpy

from src.utils import datasets
from src.agnostic import server as agnostic_server
from src.clientwise import server as clientwise_server
from src.ours import server as our_server
from src.fairfed import server as fairfed_server

from src.utils import metrics
import numpy as np
import argparse


def buildModelClass(inputSize):
    class LogisticModel(torch.nn.Module):
        def __init__(self, inputSize=inputSize, outputSize=1):
            super(LogisticModel, self).__init__()
            self.linear = torch.nn.Linear(inputSize, outputSize, bias=True)

        def forward(self, x):
            out = self.linear(x)
            return out
    return LogisticModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['agnostic', 'clientwise', 'ours', 'fairfed'])     # method name
    parser.add_argument('--home')                                                   # home folder location
    parser.add_argument('--numSeeds', default=10, type=int)                         # number of different seeds
    parser.add_argument('--numComRnds', default=100, type=int)                      # number of communcation rounds
    parser.add_argument('--numSteps', default=11, type=int)                       # number of heterogenity_values to check
    parser.add_argument('--runName', default='Synthetic_heterogenity_ablation')                        # run name for weights and biases

    args = parser.parse_args()
    HOMEFOLDER = args.home
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    method = args.method  # agnostic, clientwise, ours
    heterogenity_values = np.linspace(0.5, 1.0, args.numSteps)
    l=50
    mu = 1
    betas = [0.5, 1.0, 2.0] if method== 'fairfed' else [1]
    for beta in betas:
        for NY in [50]:
            for heterogenity in heterogenity_values:
                for seed in range(args.numSeeds):
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    if method == 'agnostic':
                        s = agnostic_server.Server(datasets.HeterogenityDataset(10,200,10, heterogenity).generate_data(), buildModelClass(11), torch.nn.BCEWithLogitsLoss(), m=None, T=args.numComRnds, lambda_=l, datasetname='Heterogenity', runname=args.runName, device=DEVICE, additional_config={'heterogenity': heterogenity, 'stepsize_':1e-2},client_stepsize=1e-2)
                    elif method == 'clientwise':
                        s = clientwise_server.Server(datasets.HeterogenityDataset(10,200,10, heterogenity).generate_data(), buildModelClass(11), torch.nn.BCEWithLogitsLoss(), m=None, T=args.numComRnds, lambda_=l, datasetname='Heterogenity', runname=args.runName, device=DEVICE, additional_config={'heterogenity': heterogenity, 'stepsize_':1e-2},client_stepsize=1e-2)
                    elif method == 'ours':
                        s = our_server.Server(datasets.HeterogenityDataset(10,200,10, heterogenity).generate_data(), buildModelClass(11), torch.nn.BCEWithLogitsLoss(), m=None, T=args.numComRnds, mu=mu, NY=NY, lambda_=l, datasetname='Heterogenity', runname=args.runName, device=DEVICE, additional_config={'heterogenity': heterogenity, 'stepsize_':1e-2},client_stepsize=1e-2)
                    elif method== 'fairfed':
                        s = fairfed_server.Server(datasets.HeterogenityDataset(10,200,10, heterogenity).generate_data(), buildModelClass(11), torch.nn.BCEWithLogitsLoss(), m=None, beta=beta, T=args.numComRnds, lambda_=l, datasetname='Heterogenity', runname=args.runName, device=DEVICE, additional_config={'heterogenity': heterogenity, 'stepsize_':1e-2},client_stepsize=1e-2)
                    else:
                        raise NotImplementedError(f'Method {method} currently not supported')
                    s.train_test_split()
                    s.sync_N()
                    if method == 'ours':
                        s.sync_Pa()
                    s.train()
                    res, weights = s.test_current_model()
                    performances = np.vstack([np.array([metrics.accuracy(p, y).cpu(), metrics.P1(p, a).cpu()]) for p, y, a in res])
                    np.save(os.path.join(HOMEFOLDER, f'results/{method}/synthettic_p_{l}_{seed}_{NY}.npy'), performances)
