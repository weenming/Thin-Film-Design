import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./designer/script/')
sys.path.append('./')
sys.path.append('./working/')

from film import TwoMaterialFilm
from spectrum import Spectrum
from design import BaseDesign

from analyze_utils.structure import plot_layer_thickness
from optimizer.adam import AdamThicknessOptimizer
from analyze_utils.make_design import get_anti_reflector_spec
import pickle


init_ots = np.arange(1, 20000, 100)
layer_numbers = np.arange(40, 200, 5)
rep_numbers = np.arange(5)


for init_ot in init_ots:

    for rep in rep_numbers:

        for layer_number in layer_numbers:
            np.random.seed(rep)
            d = np.random.random(layer_number)
            init_film = TwoMaterialFilm('SiO2_xc', 'Ta2O5_xc', 'SiO2_exp', d)
            d *= init_ot / (init_film.get_optical_thickness(700.)) # roughly 1000 nm ot
            
            target = get_anti_reflector_spec(inc=20., wls=np.linspace(390, 710, 320))
            design = BaseDesign([target], init_film)
            adam = AdamThicknessOptimizer(design.film, design.target_specs, 20000, alpha=1, show=False, record=True)
            films, losses = adam.optimize()
            fname = f'./design_anti_reflector/SiO2_Ta2O5-400to700nm-700nm/ot{init_ot}_layer{layer_number}_rep{rep}_design.pkl'
            with open(fname, 'wb') as f:
                pickle.dump(design, f)