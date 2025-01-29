from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import wandb
from . import metrics


def generate_split(datasets):
    trainsets = {'X': [], 'Y': [], 'A': []}
    testsets = {'X': [], 'Y': [], 'A': []}
    for ds in datasets:
        X_train, X_test, y_train, y_test, a_train, a_test = train_test_split(torch.tensor(ds[0].to_numpy()), torch.tensor(ds[1].to_numpy()), torch.tensor(ds[2].to_numpy()), test_size=0.25)
        trainsets['X'].append(X_train.float())
        trainsets['Y'].append(y_train.float())
        trainsets['A'].append(a_train.float())
        testsets['X'].append(X_test.float())
        testsets['Y'].append(y_test.float())
        testsets['A'].append(a_test.float())
    return torch.cat(trainsets['X']), torch.cat(testsets['X']), torch.cat(trainsets['Y']), torch.cat(testsets['Y']), torch.cat(trainsets['A']), torch.cat(testsets['A'])


def distance_kernel(a, b):
    return ((torch.abs(a - 1) + torch.abs(b - 1) - torch.abs(a - b)) + (torch.abs(a) + torch.abs(b) - torch.abs(a - b))) / 4


def MMD(a, b):
    return (
        distance_kernel(a, a.T).mean()
        + distance_kernel(b, b.T).mean()
        - 2 * distance_kernel(a, b.T).mean()
    )


class Trainer:
    def __init__(self, datasets, modelclass, lossf, N_epochs=1000, stepsize=5e-2, lambda_=1, datasetname='None', runname='', device='cpu', batchsize=None, additional_config={}):
        self.X, self.X_test, self.Y, self.Y_test, self.A, self.A_test = generate_split(datasets)
        self.device = device
        self.lambda_ = lambda_
        self.N_epochs = N_epochs
        self.model = modelclass().to(device)
        self.lossf = lossf
        self.stepsize = stepsize
        self.batchsize = batchsize
        config = {
            "N_epochs": N_epochs,
            "lambda_": lambda_,
            "dataset": datasetname,
            "runname": runname,
            "algorithm": "centralized",
        }
        config.update(additional_config)

        wandb.init(
            # set the wandb project where this run will be logged
            project="fairFL",

            # track hyperparameters and run metadata
            config=config
        )

    def train(self):
        if self.batchsize is not None:
            generator = DataLoader(TensorDataset(self.X, self.Y, self.A), batch_size=self.batchsize, shuffle=True, num_workers=0)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.stepsize)
        for e in range(self.N_epochs):
            # training steps
            if self.batchsize is not None:
                for x, y, a in generator:
                    optimizer.zero_grad()
                    prediction = self.model(x)
                    accloss = self.lossf(prediction.flatten(), y)
                    fairloss = MMD(prediction[a == 0], prediction[a == 1])
                    loss = accloss + 2 * self.lambda_ * fairloss
                    loss.backward()
                    optimizer.step()
            else:
                optimizer.zero_grad()
                prediction = self.model(self.X)
                accloss = self.lossf(prediction.flatten(), self.Y)
                fairloss = MMD(prediction[self.A == 0], prediction[self.A == 1])
                loss = accloss + 2 * self.lambda_ * fairloss
                loss.backward()
                optimizer.step()

            # testing
            self.log_progress()
        wandb.finish()

    def log_progress(self):
        with torch.no_grad():
            prediction = self.model(self.X_test)
            acc = metrics.accuracy(prediction.flatten(), self.Y_test.flatten())
            fairness = metrics.P1(prediction.flatten(), self.A_test.flatten())
        # print(f"ACC: {acc:.3f}  -  FAIR: {fairness:.3f}")
        wandb.log({"acc": acc.cpu(), "fairness": fairness.cpu()})
