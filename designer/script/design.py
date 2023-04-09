import numpy as np
import copy

import optimizer.needle_insert as insert
import optimizer.LM_gradient_descent as gd
from film import FilmSimple
from spectrum import SpectrumSimple
import utils.loss

class Design:
    def __init__(
        self, 
        target_specs: list[SpectrumSimple],
        init_film: FilmSimple, 
        film: FilmSimple= None
    ):
        self.init_film = copy.deepcopy(init_film) # save in case of aliasing
        self.film = film if film is not None else init_film
        self.target_specs = target_specs
        self.loss = None  # Diff of the spec between designed film and target
        self.training_films: list[FilmSimple] = [] # should use training_info
        self.training_info: \
            list[dict['loss': float, 'step': int, 'insert_gd': float, 'film': FilmSimple]] = []

    def calculate_loss(self) -> float:
        """
        Calculate the RMS wrt the target spectrum specified in self.target_film
        """
        self.loss = utils.loss.calculate_RMS_f_spec(self.film, self.target_specs)
        return self.loss

    def get_init_ot(self, wl=750.):
        assert self.init_film is not None, "undifined initial film!"
        return self.init_film.get_optical_thickness(wl)

    def get_current_ot(self, wl=750.):
        return self.film.get_optical_thickness(wl)

        
    def get_init_gt(self):
        assert self.init_film is not None, "undifined initial film!"
        return self.init_film.get_d().sum()

    def get_current_gt(self):
        return self.film.get_d().sum()

        
    def TFNN_train(
        self,
        needle_epoch, 
        record=False, 
        error=1e-5, 
        max_step=1000, 
        show=False, 
        show_warning=True, 
    ):
        """
        Combination of needle insertion and gradient descent

        "record" functionality to investigate training process
        but record may decrease performance
        """
        # preparing

        # hyperparameter: exit condition of gd
        # error = 1e-5
        # max_step = 1000

        for i in range(needle_epoch):
            # LM gradient descent
            try:
                step_count = gd.LM_optimize_d_simple(
                    self.film,
                    self.target_specs,
                    error,
                    max_step
                )
                if show:
                    print(f'{i}-th iteration, loss: {self.calculate_loss()},', \
                        f'{step_count} gd steps')
            except OverflowError as e:
                if show_warning:
                    print('Parameter overflow in gd, end needle design: \n', e.args)
                return

            # record
            if record:
                self.training_info.append({
                    'loss': self.calculate_loss(),
                    'film': copy.deepcopy(self.film),
                    'step': step_count, # gd steps in this needle iteration
                })
                if show:
                    print(f'{i}-th iteration recorded')

            # Needle insertion
            try:
                inserted, insert_grad = insert.insert_1_layer(
                    self.film,
                    self.target_specs, 
                    show=show, 
                )
                if show:
                    print(f'{i}-th iteration, new layer inserted. now ' +\
                          f'{self.film.get_layer_number()} layers')
            except OverflowError as e:
                if show_warning:
                    print('Parameter overflow in insertion, end needle design:\n', e.args)
                return # end design when problems like too many layers etc. occur
            except ValueError as e:
                if show:
                    print(f'{i}-th iteration, cannot insert.')
                if show_warning:
                    print(e.args)
                return


class DesignSimple(Design):
    """

    """

    def __init__(self,
                 target_film: FilmSimple,
                 init_film: FilmSimple = None,
                 film: FilmSimple = None
                 ):
        
        self.target_film = target_film
        if len(self.target_film.spectrum) == 0:
            raise ValueError("target_film must have nonempty spectrum")
        self.target_film.calculate_spectrum()
        target_specs = self.target_film.get_all_spec_list()

        super().__init__(target_specs, init_film, film)

    def get_target_ot(self, wl=750., correct_sub=True):
        assert self.target_film is not None, "undifined target film!"
        f = self.target_film
        # if the layer has the same material as the substrate, should not take into account
        neglect_last = correct_sub and \
                    ((f.get_n_A == f.get_n_sub and f.get_layer_number() % 2 == 1) or \
                    (f.get_n_B == f.get_n_sub and f.get_layer_number() % 2 == 0))
        return f.get_optical_thickness(wl, neglect_last_layer=neglect_last)

    def get_target_gt(self, wl=750.):
        assert self.target_film is not None, "undifined target film!"
        return self.target_film.get_d().sum()


    def get_current_ot_ratio(self, wl=750.):
        assert self.target_film is not None, "undifined target film!"
        return self.get_current_ot(wl) / self.target_film.get_optical_thickness(wl)


class DesignForSpecSimple(Design):
    '''One incidence angle, no absorption (thus only R spec)'''
    def __init__(self, target_spec: SpectrumSimple, init_film: FilmSimple, film: FilmSimple=None):
        super().__init__([target_spec], init_film, film)
