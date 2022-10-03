import numpy as np
import time
import gets.get_jacobi
import gets.get_spectrum
import gets.archive_get_jacobi
import gets.archive_get_spectrum


# 20 layers
d = np.zeros(61) + 100
materials = np.array(['SiO2' if i % 2 == 0 else 'TiO2' for i in range(61)])

start = time.time()
for test_count in range(1):
    spec1 = gets.get_jacobi.get_jacobi(np.linspace(1000, 2000, 1000), d, materials)
end = time.time()
print(end - start)


