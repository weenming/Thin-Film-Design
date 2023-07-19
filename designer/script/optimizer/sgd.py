import sys
sys.path.append('./designer/script/')


from tmm.get_jacobi_n_adjoint import get_jacobi_free_form
from tmm.get_jacobi import get_jacobi_simple
from tmm.get_spectrum import get_spectrum_free, get_spectrum_simple

from optimizer.grad_helper import stack_f, stack_J, stack_init_params
from utils.loss import calculate_RMS_f_spec, rms
from spectrum import BaseSpectrum
from film import FreeFormFilm, TwoMaterialFilm
import numpy as np
from typing import Sequence
import copy
from optimizer.optimizer import GradientOptimizer

class SGDOptimizer(GradientOptimizer):

    def __init__(
        self,
        film,
        target_spec_ls: Sequence[BaseSpectrum],
        max_steps,
        **kwargs
    ):
        super().__init__(film, target_spec_ls, max_steps, **kwargs)
        self.lr = 0.01 if 'lr' not in kwargs else kwargs['lr']
        self.mu = 0 if 'mu' not in kwargs else kwargs['mu'] # momentum
        self.tau = 0 if 'tau' not in kwargs else kwargs['tau'] # dampening
        self.nesterov = False if 'nesterov' in kwargs else kwargs['nesterov']

        # initialize optimizer
        self.max_steps = max_steps
        self.max_patience = self.max_steps if 'patience' not in kwargs else kwargs[
            'patience']
        self.current_patience = self.max_patience
        self.best_loss = 0.
        self.n_arrs_ls = stack_init_params(self.film, self.target_spec_ls)
        self.b = 0.

        self._get_param()  # init variable x

        # allocate space for f and J
        self.J = np.empty((self.total_wl_num, self.x.shape[0]))
        self.f = np.empty(self.total_wl_num)
    
    def optimize(self):
        # in case not do_record, return [initial film], [initial loss]
        self._record()

        for self.i in range(self.max_steps):
            self._optimize_step()
            self._set_param()
            if self.is_recorded:
                self._record()
            if self.is_shown:
                self._show()
            if not self._update_best_and_patience():
                break
        self.x = self.best_x
        self._set_param()  # restore to best x
        return self._rearrange_record()

    def _validate_loss(self):
        # return rms(self.f) 
        return calculate_RMS_f_spec(self.film, self.target_spec_ls)

    def _optimize_step(self):
        self._mini_batching()  # make sgd params
        stack_f(
            self.f,
            self.n_arrs_ls,
            self.film.get_d(),
            self.target_spec_ls,
            spec_batch_idx=self.spec_batch_idx,
            wl_batch_idx=self.wl_batch_idx,
            get_f=self.get_f
        )
        stack_J(
            self.J,
            self.n_arrs_ls,
            self.film.get_d(),
            self.target_spec_ls,
            MAX_LAYER_NUMBER=250,  # TODO: refactor. This is not used
            spec_batch_idx=self.spec_batch_idx,
            wl_batch_idx=self.wl_batch_idx,
            get_J=self.get_J
        )

        self.g = self.J.T @ self.f
        self.b = self.mu * self.b + (1 - self.tau) * self.g
        if self.nesterov:
            self.g = self.g + self.mu * self.b
        else:
            self.g = self.b

        self.x -= self.lr * self.g


class SGDThicknessOptimizer(SGDOptimizer):
    def __init__(
            self,
            film,
            target_spec_ls: Sequence[BaseSpectrum],
            max_steps,
            lr=1,
            **kwargs
    ):
        
        super().__init__(film, target_spec_ls, max_steps, lr=lr, **kwargs)
        self.get_f = get_spectrum_simple
        self.get_J = get_jacobi_simple

    def _set_param(self):
        # Project back to feasible domain
        self.x[self.x < 0] = 0.
        self.film.update_d(self.x)

    def _get_param(self):
        self.x = self.film.get_d()