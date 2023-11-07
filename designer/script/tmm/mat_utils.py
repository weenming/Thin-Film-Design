from numba import cuda
import cmath

@cuda.jit
def mul_right(mat1, mat2):
    """
    Multiply two 2 * 2 matrices and SAVE TO THE FIRST MATRIX!
    mat1 = mat1 @ mat2
    """
    a00 = mat1[0, 0] * mat2[0, 0] + mat1[0, 1] * mat2[1, 0]
    a01 = mat1[0, 0] * mat2[0, 1] + mat1[0, 1] * mat2[1, 1]
    a10 = mat1[1, 0] * mat2[0, 0] + mat1[1, 1] * mat2[1, 0]
    a11 = mat1[1, 0] * mat2[0, 1] + mat1[1, 1] * mat2[1, 1]

    mat1[0, 0] = a00
    mat1[0, 1] = a01
    mat1[1, 0] = a10
    mat1[1, 1] = a11
    
@cuda.jit
def mul_left(mat1, mat2):
    """
    Multiply two 2 * 2 matrices and SAVE TO THE SECOND MATRIX!
    mat2 = mat1 @ mat2
    """
    a00 = mat1[0, 0] * mat2[0, 0] + mat1[0, 1] * mat2[1, 0]
    a01 = mat1[0, 0] * mat2[0, 1] + mat1[0, 1] * mat2[1, 1]
    a10 = mat1[1, 0] * mat2[0, 0] + mat1[1, 1] * mat2[1, 0]
    a11 = mat1[1, 0] * mat2[0, 1] + mat1[1, 1] * mat2[1, 1]

    mat2[0, 0] = a00
    mat2[0, 1] = a01
    mat2[1, 0] = a10
    mat2[1, 1] = a11

@cuda.jit
def mul_to(mat1, mat2, dest):
    """
    Multiply two 2 * 2 matrices (mat1 @ mat2) and save to dest
    """
    a00 = mat1[0, 0] * mat2[0, 0] + mat1[0, 1] * mat2[1, 0]
    a01 = mat1[0, 0] * mat2[0, 1] + mat1[0, 1] * mat2[1, 1]
    a10 = mat1[1, 0] * mat2[0, 0] + mat1[1, 1] * mat2[1, 0]
    a11 = mat1[1, 0] * mat2[0, 1] + mat1[1, 1] * mat2[1, 1]

    dest[0, 0] = a00
    dest[0, 1] = a01
    dest[1, 0] = a10
    dest[1, 1] = a11


@cuda.jit
def hadm_mul(mat1, mat2):
    """
    Element-wise product, or Hadamard product of two 2 * 2 matrices
    """
    return mat1[0, 0] * mat2[0, 0] + mat1[0, 1] * mat2[0, 1] + \
        mat1[1, 0] * mat2[1, 0] + mat1[1, 1] * mat2[1, 1]


@cuda.jit
def tsp(mat, dest):
    """
    Transpose 2 * 2 matrix mat and save to dest
    """
    dest[0, 0] = mat[0, 0]
    dest[0, 1] = mat[1, 0]
    dest[1, 0] = mat[0, 1]
    dest[1, 1] = mat[1, 1]



@cuda.jit
def calc_M(Ms, Mp, n_inc, inc_ang, ni, di, wl):

    costheta = cmath.sqrt(
        1 - ((n_inc / ni) * cmath.sin(inc_ang)) ** 2)
    phi = 2 * cmath.pi * 1j * costheta * ni * di / wl
    coshi = cmath.cosh(phi)
    sinhi = cmath.sinh(phi)

    Ms[0, 0] = coshi
    Ms[0, 1] = sinhi / costheta / ni
    Ms[1, 0] = costheta * ni * sinhi
    Ms[1, 1] = coshi

    Mp[0, 0] = coshi
    Mp[0, 1] = sinhi * ni / costheta
    Mp[1, 0] = costheta / ni * sinhi
    Mp[1, 1] = coshi


@cuda.jit
def calc_M_inv(Ms, Mp, n_inc, inc_ang, ni, di, wl):
    costheta = cmath.sqrt(
        1 - ((n_inc / ni) * cmath.sin(inc_ang)) ** 2)

    phi = 2 * cmath.pi * 1j * costheta * ni * di / wl
    coshi = cmath.cosh(phi)
    sinhi = cmath.sinh(phi)

    Ms[0, 0] = coshi
    Ms[0, 1] = -sinhi / costheta / ni
    Ms[1, 0] = -costheta * ni * sinhi
    Ms[1, 1] = coshi

    Mp[0, 0] = coshi
    Mp[0, 1] = -sinhi * ni / costheta
    Mp[1, 0] = -costheta / ni * sinhi
    Mp[1, 1] = coshi


@cuda.jit
def fill_arr(A, a00, a01, a10, a11):
    A[0, 0] = a00
    A[0, 1] = a01
    A[1, 0] = a10
    A[1, 1] = a11