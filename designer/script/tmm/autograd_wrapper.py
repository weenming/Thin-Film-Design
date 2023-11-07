import torch

from get_jacobi_arb_adjoint import get_jacobi_E_free_form
from get_E import get_E_free


def get_jacobi_warpper(E_to_out, device='cpu', mode='d'):
    if mode == 'd':
        
    def get_jacobi_x_free_form(*args, **kwargs):
        jacobi_E = get_jacobi_E_free_form(*args, **kwargs)
        
        E = torch.zeros(args[0].shape[0], device=device)
        get_E_free(E, *args, **kwargs)
        # BUG: probably wrong here
        torch.autograd.grad(E_to_out(E), E)
        return jacobi
    
    return get_jacobi_x_free_form