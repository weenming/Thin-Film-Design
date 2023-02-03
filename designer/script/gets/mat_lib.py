
def mul(mat1, mat2):
    """
    Multiply two mmatrices and save to the first one
    """
    a00 = mat1[0, 0] * mat2[0, 0] + mat1[0, 1] * mat2[1, 0]
    a01 = mat1[0, 0] * mat2[0, 1] + mat1[0, 1] * mat2[1, 1]
    a10 = mat1[1, 0] * mat2[0, 0] + mat1[1, 1] * mat2[1, 0]
    a11 = mat1[1, 0] * mat2[0, 1] + mat1[1, 1] * mat2[1, 1]

    mat1[0, 0] = a00
    mat1[0, 1] = a01
    mat1[1, 0] = a10
    mat1[1, 1] = a11

def tps(mat1):
    """
    Transpose matrix 1 and save to itself
    """