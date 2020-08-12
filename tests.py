import utils as U
import torch

def rosenbrock(x, y):
    a = torch.tensor([1.0])
    b = torch.tensor([100.0])
    return (a - x)**2 + b * (y - x**2)**2

def grad_rosenbrock(x, y):
    a = torch.tensor([1.0])
    b = torch.tensor([100.0])
    return torch.tensor([[-2 * (a - x) - 4 * b * (y - x**2) * x], [2 * b * (y - x**2)]])

def hess_rosenbrock(x, y):
    a = torch.tensor([1.0])
    b = torch.tensor([100.0])
    return torch.tensor([[2 - 4 * b * y + 12 * b * x**2, - 4 * b * x], [-4 * b * x, 2 * b]])


def test_grad():
    x0 = torch.tensor([-1.0, 1.0], requires_grad=True)
    fx = rosenbrock(*x0)

    auto_dx = U.jacobian(fx, x0)
    analytical_dx = grad_rosenbrock(*x0)

    assert(torch.norm(auto_dx - analytical_dx.T[0]) == 0)

def test_hess():
    x0 = torch.tensor([-1.0, 1.0], requires_grad=True)
    fx = rosenbrock(*x0)

    auto_ddx = U.hessian(fx, x0)
    analytical_ddx = hess_rosenbrock(*x0)

    assert(torch.norm(analytical_ddx - auto_ddx) == 0)

if __name__ == '__main__':
    test_grad()
    test_hess()
