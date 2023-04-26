import sys
sys.path.append('./designer/script/')


from tmm.get_jacobi_n_adjoint import get_jacobi_free_form, get_jacobi_simple
from tmm.get_spectrum import get_spectrum_free, get_spectrum_simple

from optimizer.grad_helper import stack_f, stack_J, stack_init_params
from utils.loss import calculate_RMS_f_spec, rms
from spectrum import BaseSpectrum
from film import FreeFormFilm, TwoMaterialFilm
import numpy as np
from typing import Sequence
import copy
from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(
        self,
        film,
        target_spec_ls: Sequence[BaseSpectrum],
    ):
        self.film = film
        self.target_spec_ls = target_spec_ls

    @abstractmethod
    def _record(self):
        raise NotImplementedError

    @abstractmethod
    def _show(self):
        raise NotImplementedError

    @abstractmethod
    def optimize(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def optimize_step(self):
        raise NotImplementedError

    def _rearrange_record(self):
        return [record_i for record_i in zip(self.records)]

    def __call__(self, **kwargs):
        return self.optimize(**kwargs)
