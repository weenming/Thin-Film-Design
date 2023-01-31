from numba import cuda
import numpy as np

@cuda.jit
def f(a):
    a = a + 1
    # determine thread position
    # do the parallelized task for the current thread (see SIMT model)

def _ref_by_adr(x:np.array):
    x += 1

def call_ref_hby_adr():
    x = np.zeros((2, 10), dtype='int32')
    print("brefore:\n", x)
    _ref_by_adr(x)
    print("after:\n", x)
    
if __name__ == "__main__":
    call_ref_hby_adr()
