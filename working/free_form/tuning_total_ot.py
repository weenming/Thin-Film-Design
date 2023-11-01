# %%
import os
import sys
sys.path.append(os.path.dirname(__file__) + '/../../')
sys.path.append(os.path.dirname(__file__) + '/..')
sys.path.append(os.path.dirname(__file__) + '/../../designer/script/')


from film import FreeFormFilm, BaseFilm, TwoMaterialFilm, EqOTFilm
from design import BaseDesign
from spectrum import BaseSpectrum, Spectrum
from optimizer.adam import AdamFreeFormOptimizer

from analyze_utils.make_design import make_edgefilter_design, make_triband_filter_design, make_reflection_design, get_minus_filter_spec

from utils.loss import calculate_RMS_f_spec
from analyze_utils.result_io import save, load
from analyze_utils.structure import plot_layer_thickness

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import copy
import pickle

for exp_i in range(3):
    # design = make_edgefilter_design()
    # design = make_reflection_design(wls=np.linspace(695, 939, 500))
    reps = 1

    n_min = 1.5
    n_max = 3

    # adam optimizer parameters
    alpha = 1
    batch_size = None
    max_steps=1000
    show = False


    def exp(n_size, each_ot, target):
        np.random.seed()
        # init_n = np.random.random(n_size) + 1.5
        init_n = np.zeros(n_size) + 2
        film = EqOTFilm(init_n, each_ot * n_size, substrate=1.5)

        adam_optimizer = AdamFreeFormOptimizer(
            film, 
            target, 
            max_steps=max_steps, 
            alpha=alpha * 1e2 / (n_size * each_ot), # 0.01 for big OT films...
            record=True, 
            show=show, 
            n_min=n_min, # 1.5
            n_max=n_max, # 2.5
            wl_batch_size=batch_size # full 
        )

        adam_optimizer.optimize()
        return calculate_RMS_f_spec(film, target), film


    # wl_min = 500
    # wl_max = 1000 # 1 / wl_max - 1 / wl_min = 1 / 1000

    # target = make_triband_filter_design().target_specs # 1 / wl_max - 1 / wl_min = 
    make_settings = [[500, 530 - i, 540 + i, 600] for i in [0, 10, 20]]
    target_wl_settings = make_settings[exp_i]
    [left_m, left, right, right_m] = target_wl_settings
    n_wl = 1000
    wls = np.linspace(left_m, right_m, n_wl)
    target = [get_minus_filter_spec(wls, left=left, right=right)]
    target_name = 'minus_filter'

    # %%
    each_ots = np.array([i for i in 10 ** np.linspace(0, 3, 50)])
    n_sizes = np.array([int(i) for i in 10 ** np.linspace(0, 3, 50)])


    films_arr_rep, best_loss_arr_rep, best_film_arr_rep = [], [], []
    for rep in range(reps):
        films_arr_rep.append([])
        best_loss_arr_rep.append([])
        best_film_arr_rep.append([])
        for i, ot in enumerate(each_ots):
            films_arr_rep[-1].append([])
            best_loss_arr_rep[-1].append([])
            best_film_arr_rep[-1].append([])
            for j, n_size in enumerate(n_sizes):
                print((i + j / n_sizes.shape[0]) / each_ots.shape[0])
                best_loss, best_film = exp(n_size, ot, target)
                films_arr_rep[-1][-1].append([])
                best_loss_arr_rep[-1][-1].append(best_loss)
                best_film_arr_rep[-1][-1].append(best_film)
    print(best_loss_arr_rep)

    save(os.path.dirname(__file__) + f'/raw_result_total_ot/free_form_params_{target_name}_target_{left_m}_{left}_{right}_{right_m}', np.array(best_loss_arr_rep), np.array(best_film_arr_rep))


