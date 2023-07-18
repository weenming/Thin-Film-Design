import numpy as np
import copy
from typing import TypedDict, Sequence

import optimizer.needle_insert as insert
import optimizer.archive.LM_gradient_descent as LM_gd
from optimizer.archive.adam_d import adam_optimize
from film import TwoMaterialFilm, BaseFilm, FreeFormFilm
from spectrum import SpectrumSimple, BaseSpectrum
import utils.loss


class TrainingInfo(TypedDict):
    loss: float
    step: int
    film: BaseFilm
    extra: dict


class BaseDesign:
    init_film: BaseFilm
    film: BaseFilm
    target_specs: Sequence[BaseSpectrum]
    training_info: list[TrainingInfo]
    loss: float

    def __init__(
        self,
        target_specs: Sequence[BaseSpectrum],
        init_film: BaseFilm = None,
        film: BaseFilm = None
    ):
        assert init_film is not None or init_film is not None
        self.init_film = copy.deepcopy(init_film)  # save in case of aliasing
        self.film = film if film is not None else init_film
        self.target_specs = target_specs
        self.loss = -1.  # Diff of the spec between designed film and target
        self.training_info = []

    def calculate_loss(self) -> float:
        """
        Calculate the RMS wrt the target spectrum specified in self.target_film
        """
        self.loss = utils.loss.calculate_RMS_f_spec(
            self.film, self.target_specs)
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


class NeedleDesign(BaseDesign):
    def needle_train(
        self,
        needle_epoch,
        record=False,
        error=1e-5,
        max_step=1000,
        show=False,
        show_warning=True,
    ):
        """
        Combination of needle insertion and LM gradient descent

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
                step_count = LM_gd.LM_optimize_d_simple(
                    self.film,
                    self.target_specs,
                    error,
                    max_step
                )
                if show:
                    print(f'{i}-th iteration, loss: {self.calculate_loss()},',
                          f'{step_count} gd steps')
            except OverflowError as e:
                if show_warning:
                    print('Parameter overflow in gd, end needle design: \n',
                          e.args)
                return

            # record
            if record:
                self.training_info.append({
                    'loss': self.calculate_loss(),
                    'film': copy.deepcopy(self.film),
                    'step': step_count,  # gd steps in this needle iteration
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
                    print(f'{i}-th iteration, new layer inserted. now ' +
                          f'{self.film.get_layer_number()} layers')
            except OverflowError as e:
                if show_warning:
                    print(
                        'Parameter overflow in insertion, end needle design:\
                            \n', e.args)
                # end design when problems like too many layers occur
                return
            except ValueError as e:
                if show:
                    print(f'{i}-th iteration, cannot insert.')
                if show_warning:
                    print(e.args)
                return


class ThicknessGradientDesign(BaseDesign):
    def adam_gd(self, step, record=True, **kwargs):
        if record:
            losses, films = adam_optimize(
                self.film,
                self.target_specs,
                step,
                record=record,
                **kwargs
            )
            for i, (loss, film) in enumerate(zip(losses, films)):
                self.training_info.append({
                    'loss': loss,
                    'film': film,
                    'step': i
                })
        else:
            adam_optimize(
                self.film,
                self.target_specs,
                step,
                record,
                **kwargs
            )


class FreeFormDesign(BaseDesign):
    def __init__(
        self,
        target_specs: Sequence[BaseSpectrum],
        init_film: BaseFilm = None,
        film: FreeFormFilm = None
    ):
        super().__init__(target_specs, init_film, film)

    def adam_gd_topological_design(self):
        return


class DesignForFilm(NeedleDesign, ThicknessGradientDesign):
    """
        Simple design target: spectrum generated by a film
    """

    def __init__(self,
                 target_film: BaseFilm,
                 init_film: BaseFilm = None,
                 film: TwoMaterialFilm = None
                 ):

        self.target_film = target_film
        if len(self.target_film.spectrums) == 0:
            raise ValueError("target_film must have nonempty spectrum")
        self.target_film.calculate_spectrum()
        target_specs = self.target_film.get_all_spec_list()

        super(DesignForFilm, self).__init__(target_specs, init_film, film)

    def get_target_ot(self, wl=750., correct_sub=True):
        assert self.target_film is not None, "undifined target film!"
        f = self.target_film
        # if the layer has the same material as the substrate,
        # should not take into account
        neglect_last = correct_sub and \
            ((f.get_n_A == f.get_n_sub and f.get_layer_number() % 2 == 1) or
             (f.get_n_B == f.get_n_sub and f.get_layer_number() % 2 == 0))
        return f.get_optical_thickness(wl, neglect_last_layer=neglect_last)

    def get_target_gt(self, wl=750.):
        assert self.target_film is not None, "undifined target film!"
        return self.target_film.get_d().sum()

    def get_current_ot_ratio(self, wl=750.):
        assert self.target_film is not None, "undifined target film!"
        return self.get_current_ot(wl) / \
            self.target_film.get_optical_thickness(wl)


class DesignForSpecSimple(NeedleDesign, ThicknessGradientDesign):
    '''One incidence angle, no absorption (thus only R spec)'''

    def __init__(
            self,
            target_specs: Sequence[BaseSpectrum],
            init_film: TwoMaterialFilm,
            film: TwoMaterialFilm = None
    ):
        super().__init__(target_specs, init_film, film)
