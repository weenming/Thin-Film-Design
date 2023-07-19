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
from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(
        self,
        film,
        target_spec_ls: Sequence[BaseSpectrum],
    ):
        self.film = film
        assert type(target_spec_ls) == list, 'must initialize optimizer with a list of Spectrums'
        self.target_spec_ls = target_spec_ls

    @abstractmethod
    def _record(self):
        '''Add information to be recorded to self.records.'''
        raise NotImplementedError

    @abstractmethod
    def _show(self):
        '''Shows the information user would like to see.'''
        raise NotImplementedError

    @abstractmethod
    def optimize(self, **kwargs):
        '''Executes the optimization process.'''
        raise NotImplementedError

    @abstractmethod
    def _optimize_step(self):
        raise NotImplementedError

    def _rearrange_record(self):
        '''Change records' first axis from iteration to info type.

        However, historically, the expected return value is  unpacked by 
        callee. The conversion is a transpose if we pretend the list to be 
        an array.
        '''
        return [record_i for record_i in zip(*self.records)]
    
    @abstractmethod
    def _validate_loss(self):
        raise NotImplementedError
    
    def __call__(self, **kwargs):
        return self.optimize(**kwargs)
    
    

class GradientOptimizer(Optimizer):
    def __init__(self, film, target_spec_ls, max_steps, **kwargs):
        super().__init__(film, target_spec_ls)

        # user functionalities
        self.is_recorded = False if 'record' not in kwargs else kwargs['record']
        self.is_shown = False if 'show' not in kwargs else kwargs['show']
        self.shown_condition = lambda x: True if 'show_condition' not in kwargs else kwargs['show_condition']
        self.records: list[list] = []

        # check batch size
        self.wl_num_min = np.min([s.WLS.shape[0] for s in target_spec_ls])
        if 'batch_size_spec' not in kwargs:
            self.batch_size_spec = len(target_spec_ls)
        else:
            self.batch_size_spec = kwargs['batch_size_wl']
        if 'batch_size_wl' not in kwargs:
            self.batch_size_wl = self.wl_num_min
        else:
            self.batch_size_wl = kwargs['batch_size_wl']
        assert self.batch_size_spec <= len(target_spec_ls) \
            and self.batch_size_wl <= self.wl_num_min  # spec with smallest wl
        self.total_wl_num = self.batch_size_wl * self.batch_size_spec * 2  # R & T


    def _update_best_and_patience(self):
        cur_loss = self._validate_loss()
        if cur_loss < self.best_loss or self.i == 0:
            self.best_loss = cur_loss
            self.best_x = copy.deepcopy(self.x)
            self.best_i = self.i
            self.current_patience = self.max_patience
        else:
            self.current_patience -= 1

        return self.current_patience > 0

    def _record(self):
        '''
        Append info of current step.

        Returned res list as: [films], [losses] since transformed
        with _rearrange_record once more before returning
        '''
        if len(self.records) != 0:
            self.records.append([
                copy.deepcopy(self.film),
                self._validate_loss()
            ])
        else:
            self.records.append([
                copy.deepcopy(self.film),
                self._validate_loss()
            ])

    def _show(self):
        if self.shown_condition(self.i):
            print(
                f'iter {self.i}, loss {self._validate_loss()}')
            
    def _mini_batching(self):
        '''
        Make mini-batches.
        mat: #wls \cross #spec; pick out elem on the crossing of
            rows=wl_idx and cols=spec_idx. For selected wl, R and T
            are calculated simultaneously.

        The size of J is fixed but the stored grads are different
            in each epoch according to the random shuffle.
        '''
        self.spec_batch_idx = np.random.default_rng().choice(
            len(self.target_spec_ls),
            self.batch_size_spec,
            replace=False
        )
        self.spec_batch_idx = np.sort(self.spec_batch_idx)

        self.wl_batch_idx = np.random.default_rng().choice(
            self.wl_num_min,
            self.batch_size_wl,
            replace=False
        )
        self.wl_batch_idx = np.sort(self.wl_batch_idx)



    @abstractmethod
    def _get_param(self):
        '''
        Initialize x from film
            (e.g. thickness vector in adam_d, refractive vector in adam_n)
            and then empty J and f arrays could be initialized
        '''
        raise NotImplementedError

    @abstractmethod
    def _set_param(self):
        '''
        Update film with current x
        Along with different projection strategies
        '''
        raise NotImplementedError
