import torch
import numpy

from .. import datasets
import server

from .. import metrics
import numpy as np


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
    NREPE = 10
    T = 100
    NLAMBDAS = 50
    ls = np.logspace(-5, 2, NLAMBDAS)
    for mu in [0.1, 0.5, 1.0]:
        for NY in [10, 100, 1000]:
            for l in ls:
                for seed in range(NREPE):
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    s = server.Server(datasets.CommunitiesCrimeDataset().load_data(), buildModelClass(99), torch.nn.BCEWithLogitsLoss(), m=None, T=T, lambda_=l, datasetname='CommunitiesCrime', runname='25Sept2023')
                    s.train_test_split()
                    s.sync_N()
                    s.train()
                    res, weights = s.test_current_model()
                    performances = np.vstack([np.array([metrics.accuracy(p, y), metrics.P1(p, a)]) for p, y, a in res])
                    np.save(f'../results/agnostic/cc_p_{l}_{seed}.npy', performances)
                    np.save(f'../results/agnostic/cc_w_{l}_{seed}.npy', weights)
