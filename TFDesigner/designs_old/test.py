import numpy as np
import time
import gets.get_jacobi
import gets.get_spectrum
import gets.archive_get_jacobi
import gets.archive_get_spectrum


# 36 layers
d = np.zeros(36) + 100
materials = np.array(['SiO2' if i % 2 == 0 else 'TiO2' for i in range(36)])

start = time.time()
for test_count in range(1):
    spec1 = gets.get_jacobi.get_jacobi(np.linspace(500, 1000, 500), d, materials)
end = time.time()
print(end - start)


