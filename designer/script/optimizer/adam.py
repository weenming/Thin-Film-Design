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
from optimizer.optimizer import Optimizer
from abc import abstractmethod


class AdamOptimizer(Optimizer):
    def __init__(
        self,
        film,
        target_spec_ls: Sequence[BaseSpectrum],
        max_steps,
        **kwargs
    ):
        super().__init__(film, target_spec_ls)

        # adam hyperparams
        self.alpha = 0.001 if 'alpha' not in kwargs else kwargs['alpha']
        self.beta1 = 0.9 if 'beta1' not in kwargs else kwargs['beta1']
        self.beta2 = 0.999 if 'beta2' not in kwargs else kwargs['beta2']
        self.epsilon = 1e-8 if 'epsilon' not in kwargs else kwargs['epsilon']

        # user functionalities
        self.is_recorded = False if 'record' not in kwargs else kwargs['record']
        self.is_shown = False if 'show' not in kwargs else kwargs['show']
        self.records = []

        if 'optimize' in kwargs:
            self.optimize = 'kwargs'['optimize']  # not sure if this is allowed

        # check batch size
        self.wl_num_min = np.min([s.WLS.shape[0] for s in target_spec_ls])
        if 'batch_size_spec' not in kwargs['batch_size_spec']:
            self.batch_size_spec = len(target_spec_ls)
        else:
            self.batch_size_spec = kwargs['batch_size_wl']
        if 'batch_size_wl' not in kwargs['batch_size_wl']:
            self.batch_size_wl = self.wl_num_min
        else:
            self.batch_size_wl = kwargs['batch_size_wl']
        assert self.batch_size_spec <= len(target_spec_ls) \
            and self.batch_size_wl <= self.wl_num_min  # spec with smallest wl
        self.total_wl_num = self.batch_size_wl * self.batch_size_spec * 2  # R & T

        # initialize optimizer
        self.max_steps = max_steps
        self.max_patience = self.max_steps if 'patience' not in kwargs else kwargs['patience']
        self.current_patience = self.max_patience
        self._get_param()
        # allocate space for f and J
        self.J = np.empty((self.total_wl_num, self.x.shape[0]))
        self.f = np.empty(self.total_wl_num)
        # in case not do_record, return an empty ls
        self._record()

    def optimize(self):
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

    def _loss(self):
        return rms(self.f)

    def _sgd(self):
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

    def _optimize_step(self):
        self._sgd()  # make sgd params
        self.f = stack_f(
            self.f,
            self.n_arrs_ls,
            self.film.get_d(),
            self.target_spec_ls,
            spec_batch_idx=self.spec_batch_idx,
            wl_batch_idx=self.wl_batch_idx,
            get_f=self.get_f
        )
        self.J = stack_J(
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
        self.m_hat = self.m / (1 - self.beta1)
        self.v_hat = self.v / (1 - self.beta2)
        self.x -= self.alpha * self.m_hat / \
            (np.sqrt(self.v_hat) + self.epsilon)

    def _update_best_and_patience(self):
        cur_loss = self._loss()
        if cur_loss < self.best_loss or self.i == 0:
            self.best_loss = cur_loss
            self.best_x = copy.deepcopy(self.x)
            self.best_i = self.i
            self.current_patience = self.max_patience
        else:
            self.current_patience -= 1

        return self.current_patience

    def _record(self):
        self.records.append([
            copy.deepcopy(self.film),
            self._loss()
        ])

    def _show(self):
        print(
            f'iter {self.i}, loss {self._loss()}')

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


class AdamThicknessOptimizer(AdamOptimizer):
    def __init__(
            self,
            film,
            target_spec_ls: Sequence[BaseSpectrum],
            max_steps,
            alpha=1,
            **kwargs
    ):
        super().__init__(
            film,
            target_spec_ls,
            max_steps,
            alpha=alpha,
            ** kwargs
        )
        # avoid grad explode by asserting no total reflection
        n_min = film.calculate_n_inc(target_spec_ls[0].WLS)[0] * \
            np.sin(target_spec_ls[0].INC_ANG) if n_min == 0 else n_min
        self.get_f = get_spectrum_free
        self.get_J = get_jacobi_free_form

    def _set_param(self):
        # Project back to feasible domain
        self.x[self.x < 0] = 0.
        self.film.update_d(self.x)

    def _get_param(self):
        self.x = self.film.get_d()


class AdamFreeFormOptimizer(AdamOptimizer):
    def __init__(
            self,
            film: FreeFormFilm,
            target_spec_ls: Sequence[BaseSpectrum],
            max_steps,
            alpha=0.1,
            **kwargs
    ):
        super().__init__(
            film,
            target_spec_ls,
            max_steps,
            alpha=alpha,
            **kwargs
        )
        self.get_f = get_spectrum_simple
        self.get_J = get_jacobi_simple

    def _set_param(self):
        # project back to feasible region
        self.x[self.x < self.n_min] = self.n_min
        self.x[self.x > self.n_max] = self.n_max
        self.film.update_n(self.x)
        for l, s in zip(self.n_arrs_ls, self.target_spec_ls):
            l[0] = self.film.calculate_n_array(s.WLS)

    def _get_param(self):
        self.x = self.film.get_n()
