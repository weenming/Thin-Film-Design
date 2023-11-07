import torch
import numpy as np

from tmm.get_jacobi_arb_adjoint import get_jacobi_E_free_form
from tmm.get_E import get_E_free


def get_jacobi_warpper(E_to_loss, device='cuda', mode='d'):
    
    if mode == 'd':
        partial_M_wrt_x = get_partial_M_wrt_d

    def get_jacobi_y_wrt_x_free_form(*args, **kwargs):
        '''args and kwargs should be consistent with get_jacobi_E_free_form
        '''

        # move to torch
        args_torch = []
        for arg in args:
            args_torch.append(torch.tensor(arg).to(device))

        jacobi_E_wrt_M = np.zeros(args[0].shape, dtype='complex128')
        get_jacobi_E_free_form(jacobi_E_wrt_M, *args[1:], **kwargs)
        jacobi_E_wrt_M = torch.tensor(jacobi_E_wrt_M, device=device)
        
        E = np.zeros((args[0].shape[0] // 2, 2), dtype='complex128')
        get_E_free(E, *args[1:], **kwargs)
        E = torch.tensor(E, device=device, requires_grad=True)

        jacobi_y_wrt_E = torch.autograd.grad(E_to_loss(E), E)[0].t() # t for correct subsequent reshape
        jacobi_M_wrt_x = partial_M_wrt_x(*args_torch, **kwargs)
        return jacobi_E_wrt_M * jacobi_M_wrt_x
        # y_E tested, E_M not tested, M_x tested
        jacobi_y_wrt_x = jacobi_y_wrt_E.reshape(-1, 1, 1, 1) * jacobi_E_wrt_M * jacobi_M_wrt_x
        jacobi_y_wrt_x = jacobi_y_wrt_x.sum((0, -1, -2))

        return jacobi_y_wrt_x
    
    return get_jacobi_y_wrt_x_free_form


def get_partial_M_wrt_d(
    jacobi,
    wls,
    d,
    n_layers,
    n_sub,
    n_inc,
    inc_ang,
):
    wls = wls.unsqueeze(-1)
    cos = torch.sqrt(
        1 - ((n_inc.unsqueeze(-1) / n_layers) * torch.sin(inc_ang)) ** 2)
    phi = 2 * np.pi * 1j * cos * n_layers * d / wls
    coshi = torch.cosh(phi)
    sinhi = torch.sinh(phi)


    jacobi = torch.zeros_like(jacobi, dtype=torch.complex128)

    s_idx = torch.hstack((torch.arange(wls.shape[0]), torch.arange(2 * wls.shape[0], 3 * wls.shape[0])))
    jacobi[s_idx, :, 0, 0] = (2 * np.pi * 1j * n_layers * cos / wls * sinhi).repeat(2, 1)
    jacobi[s_idx, :, 0, 1] = (2 * np.pi * 1j / wls * coshi).repeat(2, 1)
    jacobi[s_idx, :, 1, 0] = (2 * np.pi * 1j * n_layers.square() * cos.square() / wls * coshi).repeat(2, 1)
    jacobi[s_idx, :, 1, 1] = (2 * np.pi * 1j * n_layers * cos / wls * sinhi).repeat(2, 1)
    

    p_idx = torch.hstack((torch.arange(wls.shape[0], 2 * wls.shape[0]), torch.arange(3 * wls.shape[0], 4 * wls.shape[0])))
    jacobi[p_idx, :, 0, 0] = (2 * np.pi * 1j * n_layers * cos / wls * sinhi).repeat(2, 1)
    jacobi[p_idx, :, 0, 1] = (2 * np.pi * 1j * n_layers.square() / wls * coshi).repeat(2, 1)
    jacobi[p_idx, :, 1, 0] = (2 * np.pi * 1j * cos.square() / wls * coshi).repeat(2, 1)
    jacobi[p_idx, :, 1, 1] = (2 * np.pi * 1j * n_layers * cos / wls * sinhi).repeat(2, 1)
    
    return jacobi