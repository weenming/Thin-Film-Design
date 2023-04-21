

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


def hadm_mul(mat1, mat2):
    """
    Element-wise product, or Hadamard product of two 2 * 2 matrices
    """
    return mat1[0, 0] * mat2[0, 0] + mat1[0, 1] * mat2[0, 1] + \
        mat1[1, 0] * mat2[1, 0] + mat1[1, 1] * mat2[1, 1]


def tsp(mat, dest):
    """
    Transpose 2 * 2 matrix mat and save to dest
    """
    dest[0, 0] = mat[0, 0]
    dest[0, 1] = mat[1, 0]
    dest[1, 0] = mat[0, 1]
    dest[1, 1] = mat[1, 1]
