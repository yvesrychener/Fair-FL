# +------------------------------------------------------------+
# |   clients.py                                                |
# |   Contains code executed by each client                    |
# |   Client abstraction is done via `Client` class            |
# +------------------------------------------------------------+

# imports
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader


def distance_kernel(a, b):
    return ((torch.abs(a - 1) + torch.abs(b - 1) - torch.abs(a - b)) + (torch.abs(a) + torch.abs(b) - torch.abs(a - b))) / 4


def MMD(a, b):
    return (
        distance_kernel(a, a.T).mean()
        + distance_kernel(b, b.T).mean()
        - 2 * distance_kernel(a, b.T).mean()
    )


# Client Class
class Client:
    def __init__(self, dataset, model, lossf, stepsize=0.1, batchsize=None, epochs=10, lambda_=1):
        self.X, self.Y, self.A = torch.Tensor(dataset[0].to_numpy()), torch.Tensor(dataset[1].to_numpy()), torch.Tensor(dataset[2].to_numpy())
        self.stepsize = stepsize
        self.batchsize = batchsize
        self.epochs = epochs
        self.alphak0 = None
        self.alphak1 = None
        self.Y_test = None
        self.A_test = None
        self.A_test = None
        self.Y0 = None
        self.Y1 = None
        self.model = model
        self.lossf = lossf
        self.lambda_ = lambda_
        self.current_C = lambda p: 0

    def client_step(self, current_theta):
        if self.batchsize is not None:
            generator = DataLoader(TensorDataset(self.X, self.Y, self.A), batch_size=self.batchsize, shuffle=True, num_workers=0)
        self.model.load_state_dict(current_theta)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.stepsize)
        for e in range(self.epochs):
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
                fairloss = MMD(prediction[a == 0], prediction[a == 1])
                loss = accloss + 2 * self.lambda_ * fairloss
                loss.backward()
                optimizer.step()
        # decrease learning rate
        self.stepsize = 0.99 * self.stepsize
        return self.model.state_dict()

    def set_N(self, N):
        self.N = N

    def get_weight(self):
        return len(self.Y)

    def test_client(self, theta):
        with torch.no_grad():
            self.model.load_state_dict(theta)
            return self.model(self.X_test), self.Y_test, self.A_test

    def split_train_test(self, **kwargs):
        self.X, self.X_test, self.Y, self.Y_test, self.A, self.A_test = train_test_split(self.X, self.Y, self.A, **kwargs)
