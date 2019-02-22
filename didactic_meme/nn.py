import torch
import torch.nn as nn


class Concat(nn.Module):
    def forward(self, x):
        return torch.cat(x, dim=-1)

class Branch(nn.Module):
    def __init__(self, *branches):
        super().__init__()
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
        return tuple(map(lambda branch,input: branch(input), self.branches, x))

class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class Split(nn.Module):
    def __init__(self, *index):
        super().__init__()
        self.index = list(index)

    def forward(self, x):
        return tuple(map(lambda start,end: x[..., start:end], [0] + self.index, self.index + [x.shape[-1]]))

class Duplicate(nn.Module):
    def __init__(self, n_times=2):
        super().__init__()
        self.n_times = n_times

    def forward(self, x):
        return (x for _ in range(self.n_times))

class Identity(nn.Module):
    def forward(self, x):
        return x
