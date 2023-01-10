import numpy as np
from gets.get_n import get_n


class Film:
    def __init__(self):
        self.d = np.array([])
        self.n = np.array([], dtype='complex')


class FilmSimple(Film):
    '''
    Film that consist of 2 materials.
    '''

    def __init__(self):
        super().__init__()

    def _set_materials(self):
        return

    def _set_n(self):
        get_n()
