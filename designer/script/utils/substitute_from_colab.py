def calculate_dB(spec: SpectrumSimple, d, layer_index):
    i = layer_index
    Q1 = get_W.get_W_before_ith_layer(
        spec.WLS,
        d,
        spec.n,
        spec.n_sub,
        spec.n_inc,
        spec.INC_ANG, 
        i
    )

    Q2 = get_W.get_W_after_ith_layer(
        spec.WLS,
        d,
        spec.n,
        spec.n_sub,
        spec.n_inc,
        spec.INC_ANG, 
        i
    )

    nB = np.repeat(spec.n[:, i + 1], 2, axis=0)
    nA = np.repeat(spec.n[:, i], 2, axis=0)
    n_inc = np.repeat(spec.n_inc, 2, axis=0)
    cosA = np.sqrt(1 - ((n_inc / nA) * np.sin(spec.INC_ANG)) ** 2)
    cosB = np.sqrt(1 - ((n_inc / nB) * np.sin(spec.INC_ANG)) ** 2)
    wls = np.repeat(spec.WLS, 2, axis=0)
    dA = d[i]
    
    A_1 = np.array([[[0, 1], [0, 0]] for k in range(2 * spec.WLS.shape[0])])
    A_2 = np.array([[[0, 0], [1, 0]] for k in range(2 * spec.WLS.shape[0])])
    # solve A_lambda1 and A_lambda2 respectively and acquire the ratio between d_B and d_A
    # TODO: should have used hermite conjugate. Don't forget to check consistency in the real part!
    partialL_A_1 = np.transpose(Q1, (0, 2, 1)).conj() @ Q1 @ A_1 @ Q2 @ np.array([[1., 0.], [0., 0.]]) @ np.transpose(Q2, (0, 2, 1)).conj()
    partialL_A_2 = np.transpose(Q1, (0, 2, 1)).conj() @ Q1 @ A_2 @ Q2 @ np.array([[1., 0.], [0., 0.]]) @ np.transpose(Q2, (0, 2, 1)).conj()
    a1 = nA**2 * cosA**2 * partialL_A_1[:, 0, 1] + partialL_A_1[:, 1, 0]
    a2 = nB**2 * cosB**2 * partialL_A_2[:, 0, 1] + partialL_A_2[:, 1, 0]

    dB = dA * (np.dot(a1, (nA**2 * cosA**2 / wls)) - np.dot(a2, 1 / wls)) / \
        (np.dot(a1, (nB**2 * cosB**2 / wls)) - np.dot(a2, 1 / wls))
    
    # save substitution info
    this_ot_ratio = (nB[wls.shape[0]//2] * dB) / (nA[wls.shape[0]//2] * dA)

    return dB, this_ot_ratio
    
  
def optimal_and_thin_film_approx_substitution_onestep_new(f: FilmSimple, d_min):
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
    ratios = []
    delete_indices = []
    i = 1
    while i < d.shape[0]: # neglect the last layer
        if f.get_d()[i] < d_min:
            dB, this_ot_ratio = calculate_dB(spec, f.get_d(), i)
            count += 1
            ratios.append(this_ot_ratio)
            # update d
            i_to_add = i - 1
            for i_to_del in delete_indices[::-1]:
                if i_to_del != i_to_add:
                    break
                else:
                    i_to_add -= 1
            
            if i == d.shape[0] - 1:
                d[i_to_add] += dB
            else:
                d[i_to_add] += d[i + 1] + dB
            delete_indices += [i, i + 1]
            i += 1

        i += 1
        # end loop

    
    d = np.delete(d, delete_indices).real
    f.update_d(d)
    return count, ratios

def equal_optical_thickness(f: FilmSimple, d_min):
    d = f.get_d()
    i = 1
    count = 0
    while i < d.shape[0] - 1:
        if d[i] < d_min:
            n_arr = f.calculate_n_array(np.array([750]))
            optical_ratio = n_arr[0, i + 1] / n_arr[0, i]
            d[i - 1] += optical_ratio * d[i] + d[i + 1]
            d = np.delete(d, [i, i + 1])
            i -= 1
            count += 1
        i += 1
    f.update_d(d)
    return count

def search_ot_substitution(f: FilmSimple, f_origin: FilmSimple, d_min):
    d = f.get_d()
    i = 1
    count = 0
    ratios = []
    while i < d.shape[0] - 1:
        if d[i] < d_min:
            n_arr = f.calculate_n_array(np.array([750]))
            optical_ratio = n_arr[0, i + 1] / n_arr[0, i]
            m = calculate_RMS(f, f_origin)

            for r in np.linspace(0, d_min * 10, 10):
                d_tmp = d.copy()
                # substitute
                d_tmp[i - 1] += r * optical_ratio * d[i] + d[i + 1]
                d_tmp = np.delete(d_tmp, [i, i + 1])

                f.update_d(d_tmp)
                m = calculate_RMS(f, f_origin)
                if r == 0. or m < best_m: # first search set as tmp best
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

def eqot_vs_optim(thin_thickness, thin_number=1, thin_layer_idx=None):
    '''One layer of thin film to substitute'''
    avg, var = 100, 20
    n = 20
    if thin_layer_idx is None:
        thin_layer_idx = np.random.randint(1, n - 1, size=thin_number)

    d_org = np.array([100] * n) + np.random.random(n) * var
    d_org[thin_layer_idx] = thin_thickness
    
    f_org = FilmSimple('SiO2', 'TiO2', 'SiO2', d_org)
    f_org.add_spec_param(60., np.linspace(500, 1000, 500))

    f_eq_ot, f_optim = copy.deepcopy(f_org), copy.deepcopy(f_org)

    count_eq_ot = equal_optical_thickness(
        f_eq_ot, thin_thickness + 1e-5
    )

    count_optim, ratio = optimal_and_thin_film_approx_substitution_onestep_new(
        f_optim, thin_thickness + 1e-5
    )

    return f_org, f_eq_ot, f_optim

# RMS wrt number of thin layers
d_thins = np.linspace(0, 5, 100)
layer_numbers = np.arange(1, 10)
rep = 500

RMS_optims = []
RMS_eqots = []
for _ in range(rep):
    cur_RMS_optims = []
    cur_RMS_eqots = []
    for layer_number in layer_numbers:
        f_org, f_eq_ot, f_optim = eqot_vs_optim(
            thin_thickness = 1,
            thin_number = layer_number,
            thin_layer_idx = None
        )
        cur_RMS_optims.append(calculate_RMS(f_org, f_optim))
        cur_RMS_eqots.append(calculate_RMS(f_org, f_eq_ot))
    RMS_optims.append(cur_RMS_optims)
    RMS_eqots.append(cur_RMS_eqots)
mean_eqots, std_eqots = np.array(RMS_eqots).mean(axis=0), np.array(RMS_eqots).std(axis=0)
mean_optims, std_optims = np.array(RMS_optims).mean(axis=0), np.array(RMS_optims).std(axis=0)

fig, ax = plt.subplots(1, 1)
ax.plot(layer_numbers, mean_optims, label='optim sub', color='steelblue')
ax.fill_between(layer_numbers, mean_optims - std_optims, mean_optims + std_optims, color='steelblue', alpha=0.5)

ax.plot(layer_numbers, mean_eqots, label='eq ot sub', color='orange')
ax.fill_between(layer_numbers, mean_eqots - std_eqots, mean_eqots + std_eqots, color='orange', alpha=0.5)

ax.legend()
ax.set_xlabel('number of thin layers (1 nm)')
ax.set_ylabel('RMS w.r.t. original film')

plt.savefig("rms_wrt_thinlayer_number_colab.png", dpi=300)
files.download("rms_wrt_thinlayer_number_colab.png") 

# RMS wrt thickness of ONE thin layer
d_thins = np.linspace(1, 9, 9)
rep = 500

RMS_optims = []
RMS_eqots = []
for _ in range(rep):
    cur_RMS_optims = []
    cur_RMS_eqots = []
    for thin_thickness in d_thins:
        f_org, f_eq_ot, f_optim = eqot_vs_optim(
            thin_thickness = thin_thickness,
            thin_number = 1,
            thin_layer_idx = None
        )
        cur_RMS_optims.append(calculate_RMS(f_org, f_optim))
        cur_RMS_eqots.append(calculate_RMS(f_org, f_eq_ot))
    RMS_optims.append(cur_RMS_optims)
    RMS_eqots.append(cur_RMS_eqots)
mean_eqots, std_eqots = np.array(RMS_eqots).mean(axis=0), np.array(RMS_eqots).std(axis=0)
mean_optims, std_optims = np.array(RMS_optims).mean(axis=0), np.array(RMS_optims).std(axis=0)

fig, ax = plt.subplots(1, 1)
ax.plot(d_thins, mean_optims, label='optim sub', color='steelblue')
ax.fill_between(d_thins, mean_optims - std_optims, mean_optims + std_optims, color='steelblue', alpha=0.5)

ax.plot(d_thins, mean_eqots, label='eq ot sub', color='orange')
ax.fill_between(d_thins, mean_eqots - std_eqots, mean_eqots + std_eqots, color='orange', alpha=0.5)

ax.legend()
ax.set_xlabel('thickness of the thin layer')
ax.set_ylabel('RMS w.r.t. original film')

plt.savefig("rms_wrt_thinlayer_thickness_colab.png", dpi=300)
files.download("rms_wrt_thinlayer_thickness_colab.png") 