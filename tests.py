import utils as U
import torch
import Math.vector3 as V3

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

def test_vector3_zero():
    xex = torch.zeros((3,))
    x = V3.zero()

    assert(torch.norm(xex - x) == 0)

def test_vector3_make():
    xex = torch.tensor([1.0, 2.0, 3.0])
    x = V3.make(1, 2, 3)

    assert(torch.norm(xex - x) == 0)

def test_vector3_i():
    xex = torch.tensor([1.0, 0.0, 0.0])
    x = V3.i()

    assert(torch.norm(xex - x) == 0)

def test_vector3_j():
    xex = torch.tensor([0.0, 1.0, 0.0])
    x = V3.j()

    assert(torch.norm(xex - x) == 0)

def test_vector3_k():
    xex = torch.tensor([0.0, 0.0, 1.0])
    x = V3.k()

    assert(torch.norm(xex - x) == 0)

def test_vector3_cross():
    at = torch.tensor([1.0, 2.0, 3.0])
    bt = torch.tensor([3.0, 2.0, 1.0])
    ct = V3.cross(at, bt)

    import numpy as np
    anp = np.array([1.0, 2.0, 3.0])
    bnp = np.array([3.0, 2.0, 1.0])
    cnp = np.cross(anp, bnp)

    assert(torch.norm(ct - torch.from_numpy(cnp)) == 0)

def test_vector3_norm():
    at = torch.tensor([1.0, 2.0, 3.0])
    import numpy as np
    anp = np.array([1.0, 2.0, 3.0])

    assert(V3.norm(at) - np.linalg.norm(anp) == 0)

if __name__ == '__main__':
    test_grad()
    test_hess()
    test_vector3_zero()
    test_vector3_make()
    test_vector3_i()
    test_vector3_j()
    test_vector3_k()
    test_vector3_cross()
    test_vector3_norm()
