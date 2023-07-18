import sys
sys.path.append('./designer/script/')

import numpy as np
import copy

from film import TwoMaterialFilm
from spectrum import BaseSpectrum
from optimizer.archive.LM_gradient_descent import stack_f, stack_J


def genetic(film: TwoMaterialFilm):
    raise NotImplementedError
