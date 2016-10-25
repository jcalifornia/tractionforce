
"""
Modified from tv method in cvxpy
"""

from cvxpy.expressions.expression import Expression
from cvxpy.atoms.norm import norm as cvxnorm
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
    return sum_entries(cvxnorm(stack, p='fro', axis=0))

def tvnorm_trace_2d(value1, value2,  Dx, Dy):
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
    value1 = Expression.cast_to_const(value1)
    value2 = Expression.cast_to_const(value2)
    len = value1.size[0]

    diffs = [ Dx*(value1 + value2) , Dy*(value1+value2)]

    stack = vstack( *[reshape(diff, 1, len) for diff in diffs])
    return sum_entries(cvxnorm(stack, p='fro', axis=0))

def l2_trace_2d(value1, value2,  Dx, Dy):
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
    value1 = Expression.cast_to_const(value1)
    value2 = Expression.cast_to_const(value2)
    len = value1.size[0]

    diffs = [ Dx*(value1 + value2) , Dy*(value1+value2)]

    stack = vstack( *[reshape(diff, 1, len) for diff in diffs])
    return sum_entries(cvxnorm(stack, p=2, axis=0))

def l1_trace_2d(value1, value2):
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
    value1 = Expression.cast_to_const(value1)
    value2 = Expression.cast_to_const(value2)
    len = value1.size[0]

    return sum_entries(cvxnorm(value1+value2, p='fro'))

def tvnorm_det_2d(value1, value2,  Dx, Dy):
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
    value1 = Expression.cast_to_const(value1)
    value2 = Expression.cast_to_const(value2)
    len = value1.size[0]

    diffs = [ Dx*(value1 + value2) , Dy*(value1+value2)]

    stack = vstack( *[reshape(diff, 1, len) for diff in diffs])
    return sum_entries(cvxnorm(stack, p='fro', axis=0))