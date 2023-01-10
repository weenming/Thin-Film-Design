import numpy as np
import matplotlib.pyplot as plt
from gets.get_spectrum import get_spectrum
from gets.get_n import get_n


def calculate_optical_thickness(d, materials, wl=750):
    optical_thickness = np.dot(d, get_n(wl, materials)[1:-1, 0].real)
    return optical_thickness

# next: find 50 layer with suitable optical thickness and spec shape
def main():
    wl_pts = 1000
    wls = np.linspace(500, 1500, wl_pts)
    layer_num = 50
    fig, (reflect, fourier) = plt.subplots(1, 2)
    np.random.seed(0)
    for i in range(1):
        # generate different layers with same total optical thickness
        d = np.random.random_sample(layer_num)
        materials = np.array([])
        available_materials = np.array(['SiO2', 'TiO2'])  # these two have wrongly weak dispersions
        for j in range(layer_num):
            materials = np.append(materials, available_materials[j % 2])  # the first layer is available_materials[0]
        # intended total optical thickness = 1000 nm
        optical_thickness = 5000.
        # scale to total optical thickness
        # test_thickness = d.sum()
        test_optical_thickness = calculate_optical_thickness(d, materials)
        d = d * optical_thickness / test_optical_thickness

        spec = get_spectrum(wls, d, materials, theta0=60)[0:wl_pts, 0].copy()  # incident angle 60 degrees
        # compute fft of this spec
        freq_of_spec_uniform = np.linspace(1 / wls[0], 1 / wls[-1], 100)  # FFT should be carried on evenly sampled
        # signal: wls->1/wls, seems np.interp can not deal with decreasing x
        spec_sampled_at_uniform_freq = np.flip(np.interp(np.flip(freq_of_spec_uniform),
                                                         np.flip(1 / wls), np.flip(spec)))
        # fig2, ax = plt.subplots(1,1)
        # ax.plot(freq_of_spec_uniform, spec_sampled_at_uniform_freq)
        # fig2.show()

        fft_spec = np.abs(np.fft.fft(spec_sampled_at_uniform_freq))
        fft_freq_pts = freq_of_spec_uniform.shape[0]
        fft_freq = np.fft.fftfreq(fft_freq_pts, (1/wls[0] - 1/wls[-1])/fft_freq_pts)
        # print(fft_freq)
        # print(fft_spec)
        reflect.plot(1 / wls, spec, label=f'the {i}-th run')
        # fourier.plot(1/(np.abs(fft_freq)), fft_spec, label=f'the {i}-th run')
        fourier.scatter(np.abs(fft_freq), fft_spec, label=f'the {i}-th run', marker='x', color='0.1', alpha=0.2)  # x-axis is period (frequency in EM
        # wave's frequency)

    reflect.set_xlabel('$\\frac{1}{\\lambda}/nm^{-1}$')
    reflect.set_ylabel('$reflectance$')
    reflect.set_title('reflection')
    reflect.legend()

    fourier.set_title('FFT of specs')
    fourier.set_xlabel('$freq 1/\\frac{1}{\\lambda}/nm$')
    fourier.set_ylabel('$fft of specs')
    # fourier.legend()

    plt.show()
    fig.savefig(f'../results/total_optical_thickness{optical_thickness}.jpg', dpi=300)


if __name__ == '__main__':
    main()
