import numpy as np
import optimizer.needle_insert as insert
import optimizer.LM_gradient_descent as gd
from film import FilmSimple
import utils.loss


class DesignSimple:
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
        self.init_film = init_film
        self.target_film.calculate_spectrum()
        self.film = film
        self.loss = None  # Diff of the spec of designed film and target spec in RMS
        self.training_films: list[FilmSimple] = []
        
    def get_init_ot(self, wl=750.):
        assert self.init_film is not None, "undifined initial film!"
        return self.init_film.get_optical_thickness(wl)

    def get_current_ot(self, wl=750.):
        return self.film.get_optical_thickness(wl)

    def get_target_ot(self, wl=750., correct_sub=True):
        assert self.target_film is not None, "undifined target film!"
        f = self.target_film
        # if the layer has the same material as the substrate, should not take into account
        neglect_last = correct_sub and \
                    ((f.get_n_A == f.get_n_sub and f.get_layer_number() % 2 == 1) or \
                    (f.get_n_B == f.get_n_sub and f.get_layer_number() % 2 == 0))
        return f.get_optical_thickness(wl, neglect_last_layer=neglect_last)


    def get_init_gt(self):
        assert self.init_film is not None, "undifined initial film!"
        return self.init_film.get_d().sum()

    def get_current_gt(self):
        return self.film.get_d().sum()

    def get_target_gt(self, wl=750.):
        assert self.target_film is not None, "undifined target film!"
        return self.target_film.get_d().sum()


    def get_current_ot_ratio(self, wl=750.):
        assert self.target_film is not None, "undifined target film!"
        return self.get_current_ot(wl) / self.target_film.get_optical_thickness(wl)

    def calculate_loss(self) -> float:
        """
        Calculate the RMS wrt the target spectrum specified in self.target_film
        """
        self.loss = utils.loss.calculate_RMS(self.target_film, self.film)
        return self.loss

    def flexible_TFNN_train(self, epoch, record=False):
        """
        Combination of needle insertion and gradient descent

        TODO: "record" functionality to study early-stopping in a single training
        """
        # preparing

        # hyperparameter: exit condition of gd
        error = 1e-5
        max_step = 1000

        for i in epoch:
            # LM gradient descent
            d_new, merit_new = gd.LM_optimize_d_simple(
                self.film,
                self.target_film,
                error,
                max_step
            )

            # Needle insertion
            insert.needle_insertion(
                self.film,
                self.target_film,
            )
