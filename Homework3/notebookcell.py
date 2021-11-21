from fully_connected_networks import Linear
import torch
import eecs598
from eecs598 import reset_seed, Solver
from coutils import fix_random_seed, rel_error, compute_numeric_gradient, Solver

fix_random_seed(0)
x = torch.randn(10, 10, **to_double_cuda)
dout = torch.randn(*x.shape, **to_double_cuda)

dx_num = compute_numeric_gradient(lambda x: ReLU.forward(x)[0], x, dout)

_, cache = ReLU.forward(x)
dx = ReLU.backward(dout, cache)

# The error should be on the order of e-12
print('Testing ReLU.backward function:')
print('dx error: ', rel_error(dx_num, dx))