import numpy as np
from film import FilmSimple

def diff_simple_film(film1: FilmSimple, film2: FilmSimple):
    wl = 750. # wl at which refractive index is evaluated
    film1.get_n_A