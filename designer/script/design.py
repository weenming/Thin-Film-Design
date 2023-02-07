import numpy as np
import needle_insert as insert
import optimizer.LM_gradient_descent as gd
import film


class DesignSimple:
    """
    
    """
    def __init__(self, 
                target_film: film.FilmSimple, 
                init_film: film.FilmSimple
                ):
        self.target_film = target_film
        if len(self.target_film.spectrum) == 0:
            raise ValueError("target_film must have nonempty spectrum")
        self.target_film.calculate_spectrum()
        self.film = init_film

    def get_ot_ratio(self):
        return self.film.get_optical_thickness() / \
             self.target_film.get_optical_thickness()
    
    def calculate_merit(self):
        """
        Calculate the merit wrt the target spectrum specified in self.target_film
        """
        film.calculate_merit(self.target_film, self.film)


    def flexible_TFNN(self, epoch, record=False):
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



