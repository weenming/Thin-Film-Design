import numpy as np
from gets.get_spectrum import get_spectrum
import matplotlib.pyplot as plt

for j in range(5):
    d = np.random.random(5)
    d *= 2000/d.sum()
    materials = np.array([])
    available_materials = np.array(['SiO2', 'TiO2'])
    for i in range(d.shape[0]):
        materials = np.append(materials, available_materials[i % 2])
    print(np.hstack((d, materials)))
    wls = np.linspace(500, 1000, 500)
    plt.plot(wls, get_spectrum(wls, d, materials, theta0=0)[0:500])
    # plt.savefig(f'generated_specs\\total thickness{d.sum()}, layer number {d.shape[0]}.png')
plt.show()

