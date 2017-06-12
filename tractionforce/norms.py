
"""
Modified from tv method in cvxpy
"""

from cvxpy.expressions.expression import Expression
from cvxpy.atoms.sum_squares import sum_squares
from cvxpy.atoms.norm import norm as cvxnorm
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.atoms.affine.sum_entries import sum_entries
from cvxpy.atoms.log_det import log_det
from cvxpy.atoms.affine.diag import diag
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.elementwise.log import log
from cvxpy.atoms.elementwise.power import power
from cvxpy.expressions.expression import Expression
from cvxpy.atoms.affine.hstack import hstack
from cvxpy import pnorm


import numpy as np




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

def tvnorm_anisotropic_2d(signal, Dx,Dy):
    magnitudes = pnorm(signal,2,axis=1)
    diffs = [Dx*magnitudes, Dy*magnitudes]
    stack = vstack( *[reshape(diff, 1, magnitudes.size[0]) for diff in diffs])
    return sum_entries(pnorm(stack,2,axis=0))

def l1_aniso_2d(value1, value2):
    """\sum \sqrt{value1^2 + value2^2}
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

    return sum_entries(cvxnorm(value1+value2, p='1'))

def l1_anisotropic_2d(signal):
    return sum_entries(cvxnorm(signal,2,axis=1))

class log2(log):
    """Elementwise :math:`\log x**2`.
    """
    def numeric(self, values):
        """Returns the elementwise natural log of x.
        """
        return 2*np.log(np.abs(values[0]))

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.
        Matrix expressions are vectorized, so the gradient is a matrix.
        Args:
            values: A list of numeric values for the arguments.
        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        rows = self.args[0].size[0]*self.args[0].size[1]
        cols = self.size[0]*self.size[1]
        # Outside domain or on boundary.
        if np.min(values[0]) <= 0:
            # Non-differentiable.
            return [None]
        else:
            grad_vals = 2.0/values[0]
            return [log2.elemwise_grad_to_diag(grad_vals, rows, cols)]

    def _domain(self):
        """Returns constraints describing the domain of the node.
        """
        return []

