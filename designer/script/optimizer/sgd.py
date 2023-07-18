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
    def __init__(self, ):
        return
    
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
        # return rms(self.f) THIS IS WRONG! should calculate on val set
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
        self.m = self.beta1 * self.m + (1 - self.beta1) * self.g
        self.v = self.beta2 * self.v + (1 - self.beta2) * self.g ** 2
        self.m_hat = self.m / (1 - self.beta1 ** (self.i + 1))
        self.v_hat = self.v / (1 - self.beta2 ** (self.i + 1))
        self.x -= self.alpha * self.m_hat / \
            (np.sqrt(self.v_hat) + self.epsilon)