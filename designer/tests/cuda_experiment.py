from numba import cuda
import numpy as np
import timeit
import matplotlib.pyplot as plt
import cmath

@cuda.jit
def f(a):
    a = a + 1
    # determine thread position
    # do the parallelized task for the current thread (see SIMT model)

def _ref_by_adr(x:np.array):
    x += 1

def call_ref_by_adr():
    x = np.zeros((2, 10), dtype='int32')
    print("brefore:\n", x)
    _ref_by_adr(x)
    print("after:\n", x)

@cuda.jit()
def vec_add(a, b, res, size):
    pos = cuda.grid(1)
    if pos < size:
        # loading to device takes most of the time
        for _ in range(100):
            res[pos] = a[pos] + b[pos]


def call_vec_add(N):
    a = cuda.to_device(np.random.random(N).astype(np.complex128))
    b = cuda.to_device(np.random.random(N).astype(np.complex128)) # np.dtype seems to work anyway
    res = cuda.device_array_like(a)
    res = np.empty_like(a)
    
    vec_add[1000, 64](a, b, res, N / 2)
    # res.copy_to_host()
    print(res)

def vec_add_cpu(N):
    a = np.random.random(N)
    b = np.random.random(N)
    for _ in range(100):
        np.exp(a)

    # print(a + b)

def comp_vecopr():
    cpu_time = np.array([])
    gpu_time = np.array([])
    Ns = np.arange(10, 10000, 300)
    for N in Ns:
        cpu_time = np.append(cpu_time, timeit.timeit(f"vec_add_cpu({N})", number=10, setup="from __main__ import vec_add_cpu"))
        gpu_time = np.append(gpu_time, timeit.timeit(f"call_vec_add({N})", number=10, setup="from __main__ import call_vec_add"))
    fig, ax = plt.subplots(1, 1)
    ax.plot(Ns, cpu_time / 100, label='cpu')
    ax.plot(Ns, gpu_time / 100, label='gpu')
    
    ax.legend()
    plt.show()

if __name__ == "__main__":
    call_vec_add(100)