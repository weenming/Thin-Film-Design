import sys
sys.path.append('./designer/script')
import numpy as np
import copy

from film import TwoMaterialFilm, MultiMaterialFilm
from spectrum import SpectrumSimple
import tmm.get_intermediate_transfer_matrix as get_W
from tmm.get_E import get_E
from utils.loss import calculate_RMS


def equal_optical_thickness(f: TwoMaterialFilm, d_min):
    d = f.get_d()
    # neglect first layer...
    i = 1
    count = 0
    while i < d.shape[0] - 1:
        if d[i] < d_min:
            n_arr = f.calculate_n_array(np.array([750]))
            optical_ratio = n_arr[0, i + 1] / n_arr[0, i]
            d[i - 1] += optical_ratio * d[i] + d[i + 1]
            d = np.delete(d, [i, i + 1])
            count += 1
            if type(f) is MultiMaterialFilm:
                f.remove_layer([i, i + 1])
                f.update_d(d)
            else:
                f.update_d(d)
        else:
            i += 1

    return count

def equal_optical_thickness_new(f: TwoMaterialFilm, d_min, wl=700):
    d = f.get_d()
    # neglect first layer...
    del_idx = []
    count = 0
    i = 1
    while i < d.shape[0] - 1:
        if d[i] >= d_min:
            i += 1
            continue
        
        n_arr = f.calculate_n_array(np.array([wl]))
        optical_ratio = n_arr[0, i] / n_arr[0, i + 1]
        count += 1
        db = optical_ratio * d[i]
        
        # update d
        # determine which index to merge the thin layer into
        i_to_add = i - 1  # .....
        for i_to_del in del_idx[::-1]:
            if i_to_del != i_to_add:
                break
            else:
                i_to_add -= 1
                assert i_to_add >= 0, 'bad merge-to index'

        if i == d.shape[0] - 1:
            d[i_to_add] += db
        else:
            d[i_to_add] += d[i + 1] + db.real
        del_idx += [i, i + 1]
        i += 2
        
        
    if type(f) is MultiMaterialFilm:
        f.remove_layer(del_idx)
        f.update_d(np.delete(d, del_idx))
    else:
        f.update_d(np.delete(d, del_idx))

    return count


def search_ot_substitution(f: TwoMaterialFilm, d_min):
    f_origin = copy.deepcopy(f)
    d = f.get_d()
    i = 1
    count = 0
    ratios = []
    while i < d.shape[0] - 1:
        if d[i] < d_min:
            n_arr = f.calculate_n_array(np.array([750]))
            optical_ratio = n_arr[0, i + 1] / n_arr[0, i]
            m = calculate_RMS(f, f_origin)
            best_m = 0
            for r in np.linspace(0, d_min * 10, 10):
                d_tmp = d.copy()
                # substitute
                d_tmp[i - 1] += r * optical_ratio * d[i] + d[i + 1]
                d_tmp = np.delete(d_tmp, [i, i + 1])

                f.update_d(d_tmp)
                m = calculate_RMS(f, f_origin)
                if r == 0. or m < best_m:  # first search set as tmp best
                    best_m = m
                    best_d = d_tmp.copy()
                    best_r = r
            d = best_d
            ratios.append(best_r)
            count += 1

            i -= 1
        i += 1
    f.update_d(d)
    return count, ratios


def optimal_and_thin_film_approx_substitution_onestep_new(f: TwoMaterialFilm, d_min):
    '''
    Assuming the layers adjacent to very thin films are thin enough. Use linerization to get 
    an analytical optimal solution in the first order approximation.

    One step: thickness compensation always adds to the previous layer. When adjaecnt thin layers,
    one of their correction would be neglected
    '''
    # assume single spec
    assert len(f.get_all_spec_list()) == 1, "too many spectrums"
    # load params to the get W function
    d = f.get_d()
    spec = f.get_spec()
    count = 0
    ratios: list[float] = []
    delete_indices: list[int] = []
    i = 1
    while i < d.shape[0]:
        if f.get_d()[i] < d_min:
            dB, this_ot_ratio = calculate_dB(spec, f.get_d(), i)
            count += 1
            ratios.append(this_ot_ratio)
            # update d
            i_to_add = i - 1  # .....
            for i_to_del in delete_indices[::-1]:
                if i_to_del != i_to_add:
                    break
                else:
                    i_to_add -= 1

            if i == d.shape[0] - 1:
                d[i_to_add] += dB
            else:
                d[i_to_add] += d[i + 1] + dB.real
            delete_indices += [i, i + 1]
            i += 1

        i += 1
        # end loop
    if type(f) is MultiMaterialFilm:
        f.remove_layer(delete_indices)
        f.update_d(np.delete(d, delete_indices).real)
    else:
        d = np.delete(d, delete_indices).real
        f.update_d(d)
    return count, ratios


def calculate_dB(spec: SpectrumSimple, d, layer_index):
    i = layer_index
    wls = spec.WLS
    n = spec.film.calculate_n_array(wls)
    W1 = get_W.get_W_before_ith_layer(
        wls,
        d,
        n,
        spec.n_sub,
        spec.n_inc,
        spec.INC_ANG,
        i
    )

    W2 = get_W.get_W_after_ith_layer(
        wls,
        d,
        n,
        spec.n_sub,
        spec.n_inc,
        spec.INC_ANG,
        i
    )

    W1_s, W1_p = W1[:wls.shape[0], :, :], W1[wls.shape[0]:, :, :]
    W2_s, W2_p = W2[:wls.shape[0], :, :], W2[wls.shape[0]:, :, :]

    nB = n[:, i + 1]
    nA = n[:, i]
    n_inc = spec.n_inc
    cosA = np.sqrt(1 - ((n_inc / nA) * np.sin(spec.INC_ANG)) ** 2)
    cosB = np.sqrt(1 - ((n_inc / nB) * np.sin(spec.INC_ANG)) ** 2)
    dA = d[i]

    Q_A_p, Q_B_p, Q_A_s, Q_B_s = np.zeros((spec.WLS.shape[0], 2, 2)), np.zeros(
        (spec.WLS.shape[0], 2, 2)), np.zeros((spec.WLS.shape[0], 2, 2)), np.zeros((spec.WLS.shape[0], 2, 2))
    def fill(arr, a10, a01):
        arr[:, 1, 0] = a10
        arr[:, 0, 1] = a01
    fill(Q_A_p, cosA ** 2, nA ** 2)
    fill(Q_B_p, cosB ** 2, nB ** 2)
    fill(Q_A_s, cosA ** 2 * nA ** 2, 1 / cosA ** 2 / nA ** 2)
    fill(Q_B_s, cosB ** 2 * nB ** 2, 1 / cosB ** 2 / nB ** 2)
    
    E0 = np.tile(np.array([[0], [1]]), (wls.shape[0], 1, 1))
    # solve A_lambda1 and A_lambda2 respectively and acquire the ratio between d_B and d_A
    # TODO: should have used hermite conjugate. Don't forget to check consistency in the real part!
    # ...what was i talking about?
    partialL_A_p = W1_p @ Q_A_p @ W2_p @ E0
    partialL_B_p = W1_p @ Q_B_p @ W2_p @ E0

    partialL_A_s = W1_s @ Q_A_s @ W2_s @ E0
    partialL_B_s = W1_s @ Q_B_s @ W2_s @ E0

    ax = [0, 2, 1] # transpose axes
    upper = (partialL_B_p.conj().transpose(*ax) * partialL_A_p).sum().real + \
        (partialL_B_s.conj().transpose(*ax) * partialL_A_s).sum().real
    lower = (partialL_B_p.conj().transpose(*ax) * partialL_B_p).sum().real + \
        (partialL_B_s.conj().transpose(*ax) * partialL_B_s).sum().real

    dB = dA * upper / lower

    # save substitution info
    this_ot_ratio = (nB[wls.shape[0] // 2] * dB) / (nA[wls.shape[0] // 2] * dA)

    return dB, this_ot_ratio
