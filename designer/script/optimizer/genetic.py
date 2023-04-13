import sys
sys.path.append('./designer/script/')

import numpy as np
import copy

from film import FilmSimple
from spectrum import BaseSpectrum
from optimizer.LM_gradient_descent import stack_f, stack_J


def genetic(film: FilmSimple):
    raise NotImplementedError