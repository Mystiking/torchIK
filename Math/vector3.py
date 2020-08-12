import torch

def zero():
    return torch.zeros((3,))

def make(x : float, y : float, z : float):
    return torch.tensor([x, y, z])

def i():
    return torch.tensor([1.0, 0.0, 0.0])

def j():
    return torch.tensor([0.0, 1.0, 0.0])

def k():
    return torch.tensor([0.0, 0.0, 1.0])

def cross(a, b):
    return torch.cross(a, b)

def norm(a):
    return torch.norm(a)
