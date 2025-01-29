import torch


def accuracy(p, y):
    return (torch.sigmoid(p).round().flatten() == y).float().mean()


def P1(p, a, abs=True):
    if abs:
        return ((torch.sigmoid(p[a == 0]).round().flatten() == 1).float().mean() - (torch.sigmoid(p[a == 1]).round().flatten() == 1).float().mean()).abs()
    else:
        return ((torch.sigmoid(p[a == 0]).round().flatten() == 1).float().mean() - (torch.sigmoid(p[a == 1]).round().flatten() == 1).float().mean())
