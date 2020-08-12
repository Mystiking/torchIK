import torch

def jacobian(fx, x0):
    flat_fx  = fx.reshape(-1)
    fx_one_hot = torch.zeros(flat_fx.shape)
    jacobian = torch.zeros((flat_fx.shape[0], x0.shape[0]))
    for i in range(flat_fx.shape[0]):
        fx_one_hot[i] = 1.0
        grad_x, = torch.autograd.grad(flat_fx, x0, fx_one_hot, retain_graph=True, create_graph=True)
        fx_one_hot[i] = 0.0
        jacobian[i, :] = grad_x.reshape(x0.shape)
    return jacobian

def hessian(fx, x0):
    return jacobian(jacobian(fx, x0), x0)
