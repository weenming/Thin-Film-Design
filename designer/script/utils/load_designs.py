from design import DesignSimple
from film import FilmSimple
import numpy as np
# Load trained films data and construct Design objects


def load_designs_single_spec(file_dir,
                            f_target: FilmSimple,
                            n_A,
                            n_B,
                            n_sub,
                            load_training_process=False
                        ):
    INC_ANG = f_target.get_spec().INC_ANG
    WLS = f_target.get_spec().WLS
    designs: list[DesignSimple] = []
    for run_idx in range(200):
        try:
            try:
                design = DesignSimple(f_target,
                                    FilmSimple(n_A,
                                                n_B,
                                                n_sub,
                                                np.loadtxt(file_dir + f'run_{run_idx}/init', dtype=float)*1000.
                                                ), # init film
                                    FilmSimple(n_A, 
                                                n_B,
                                                n_sub,
                                                np.loadtxt(file_dir + f'run_{run_idx}/final', dtype=float)*1000.
                                                )  # final film
                                    )
            except Exception as e: # old experiments do not have init rec record
                design = DesignSimple(f_target,
                    FilmSimple(n_A,
                                n_B,
                                n_sub,
                                np.loadtxt(file_dir + f'run_{run_idx}/iter_0', dtype=float)*1000.
                                ), # init film
                    FilmSimple(n_A, 
                                n_B,
                                n_sub,
                                np.loadtxt(file_dir + f'run_{run_idx}/final', dtype=float)*1000.
                                )  # final film
                    )
            if design.get_current_gt() == 0:
                raise ValueError("Trained film should not have zero geometric thickness")
            # fill params of the loaded film
            design.film.add_spec_param(inc_ang=INC_ANG, wls=WLS)
            design.film.calculate_spectrum()
            design.calculate_loss()

            designs.append(design)
        except Exception as e:
            print(e, f"(run {run_idx})")
            continue # skip invalid d
    # load design process if asked to
    if load_training_process:
        _load_design_process(designs, file_dir, n_A, n_B, n_sub)

    return designs

def init_film_single_spec(f: FilmSimple, inc, wls):
    f.add_spec_param(inc, wls)
    spec_target = f.get_spec()
    spec_target.calculate() # compile cuda kernel func

def _load_design_process(designs: list[DesignSimple], fpath:str, n_A:str, n_B: str, n_sub: str):
    design_idx = 0
    for design in designs:
        design.training_films: list[FilmSimple] = []
        for i in range(50):
            try:
                film = FilmSimple(n_A, n_B, n_sub, np.loadtxt(fpath + f'run_{design_idx}/iter_{i}') * 1000.)
                film.add_spec_param(design.target_film.get_spec().INC_ANG, design.target_film.get_spec().WLS)
                design.training_films.append(film)
            except FileNotFoundError as e:
                print(e.filename, "not found")
                continue # neglect
        design_idx += 1