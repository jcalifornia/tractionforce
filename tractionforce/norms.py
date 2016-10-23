
"""
Modified from tv method in cvxpy
"""

from cvxpy.expressions.expression import Expression
from cvxpy.atoms.norm import norm
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.atoms.affine.sum_entries import sum_entries
from cvxpy.atoms.affine.reshape import reshape


def tvnorm2d(value,  Dx, Dy):
    """Total variation of a vector, matrix, or list of matrices.
    Uses L1 norm of discrete gradients for vectors and
    L2 norm of discrete gradients for matrices.
    Parameters
    ----------
    value : Expression or numeric constant
        The value to take the total variation of.
    Returns
    -------
    Expression
        An Expression representing the total variation.
    """
    value = Expression.cast_to_const(value)
    len = value.size[0]

    diffs = [ Dx*value , Dy*value]

    stack = vstack( *[reshape(diff, 1, len) for diff in diffs])
    return sum_entries(norm(stack, p='fro', axis=0))

def detnorm(value, *args):
    pass

def tv(value, *args):
    """Total variation of a vector, matrix, or list of matrices.
    Uses L1 norm of discrete gradients for vectors and
    L2 norm of discrete gradients for matrices.
    Parameters
    ----------
    value : Expression or numeric constant
        The value to take the total variation of.
    args : Matrix constants/expressions
        Additional matrices extending the third dimension of value.
    Returns
    -------
    Expression
        An Expression representing the total variation.
    """
    value = Expression.cast_to_const(value)
    rows, cols = value.size
    if value.is_scalar():
        raise ValueError("tv cannot take a scalar argument.")
    # L1 norm for vectors.
    elif value.is_vector():
        return norm(value[1:] - value[0:max(rows, cols)-1], 1)
    # L2 norm for matrices.
    else:
        args = map(Expression.cast_to_const, args)
        values = [value] + list(args)
        diffs = []
        for mat in values:
            diffs += [
                mat[0:rows-1, 1:cols] - mat[0:rows-1, 0:cols-1],
                mat[1:rows, 0:cols-1] - mat[0:rows-1, 0:cols-1],
            ]
        length = diffs[0].size[0]*diffs[1].size[1]
        stacked = vstack(*[reshape(diff, 1, length) for diff in diffs])
        return sum_entries(norm(stacked, p='fro', axis=0))