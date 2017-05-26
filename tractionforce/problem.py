import numpy as np
import cvxpy
from cvxpy import Variable, Minimize, sum_squares, norm, Problem, Parameter, mul_elemwise, sum_entries, Constant
from scipy import sparse
import sys

from .elasticity import *


def gen_matrices(x_in, y_in, x_out, y_out, dx, dy, loworder = False):
    x_center = np.mean(x_in)
    y_center = np.mean(y_in)
    n_in = len(x_in)
    n_out = len(x_out)

    print("Size of the problem is " + str(n_in + n_out))

    deltax_in_in = x_in[..., np.newaxis] - x_in[np.newaxis, ...]  # should be x-x'
    deltax_out_in = x_out[..., np.newaxis] - x_in[np.newaxis, ...]  # should be x-x'
    deltay_in_in = y_in[..., np.newaxis] - y_in[np.newaxis, ...]  # y - y'
    deltay_out_in = y_out[..., np.newaxis] - y_in[np.newaxis, ...]  # y - y'

    l2_in_plus_in_plus = (np.array([deltax_in_in * dx - dx / 2.0, deltay_in_in * dy - dy / 2.0]) ** 2).sum(
        axis=0) ** 0.5
    l2_in_plus_in_minus = (np.array([deltax_in_in * dx - dx / 2.0, deltay_in_in * dy + dy / 2.0]) ** 2).sum(
        axis=0) ** 0.5
    l2_in_minus_in_plus = (np.array([deltax_in_in * dx + dx / 2.0, deltay_in_in * dy - dy / 2.0]) ** 2).sum(
        axis=0) ** 0.5
    l2_in_minus_in_minus = (np.array([deltax_in_in * dx + dx / 2.0, deltay_in_in * dy + dy / 2.0]) ** 2).sum(
        axis=0) ** 0.5

    l2_out_plus_in_plus = (np.array([deltax_out_in * dx - dx / 2.0, deltay_out_in * dy - dy / 2.0]) ** 2).sum(
        axis=0) ** 0.5
    l2_out_plus_in_minus = (np.array([deltax_out_in * dx - dx / 2.0, deltay_out_in * dy + dy / 2.0]) ** 2).sum(
        axis=0) ** 0.5
    l2_out_minus_in_plus = (np.array([deltax_out_in * dx + dx / 2.0, deltay_out_in * dy - dy / 2.0]) ** 2).sum(
        axis=0) ** 0.5
    l2_out_minus_in_minus = (np.array([deltax_out_in * dx + dx / 2.0, deltay_out_in * dy + dy / 2.0]) ** 2).sum(
        axis=0) ** 0.5

    x_adjacency = sparse.csr_matrix(
        (deltax_in_in == -1) * (deltay_in_in == 0) * -1 + (deltax_in_in == 1) * (deltay_in_in == 0) * 1)
    y_adjacency = sparse.csr_matrix(
        (deltay_in_in == -1) * (deltax_in_in == 0) * -1 + (deltay_in_in == 1) * (deltax_in_in == 0) * 1)

    A_in_in_x = fxx(deltax_in_in * dx - dx / 2., deltay_in_in * dy - dy / 2.0, l2_in_plus_in_plus) - \
                fxx(deltax_in_in * dx - dx / 2., deltay_in_in * dy + dy / 2.0, l2_in_plus_in_minus) - \
                fxx(deltax_in_in * dx + dx / 2., deltay_in_in * dy - dy / 2.0, l2_in_minus_in_plus) + \
                fxx(deltax_in_in * dx + dx / 2., deltay_in_in * dy + dy / 2.0, l2_in_minus_in_minus)

    A_out_in_x = fxx(deltax_out_in * dx - dx / 2., deltay_out_in * dy - dy / 2.0, l2_out_plus_in_plus) - \
                 fxx(deltax_out_in * dx - dx / 2., deltay_out_in * dy + dy / 2.0, l2_out_plus_in_minus) - \
                 fxx(deltax_out_in * dx + dx / 2., deltay_out_in * dy - dy / 2.0, l2_out_minus_in_plus) + \
                 fxx(deltax_out_in * dx + dx / 2., deltay_out_in * dy + dy / 2.0, l2_out_minus_in_minus)

    D_in_in_x = fxy(deltax_in_in * dx - dx / 2., deltay_in_in * dy - dy / 2.0, l2_in_plus_in_plus) - \
                fxy(deltax_in_in * dx - dx / 2., deltay_in_in * dy + dy / 2.0, l2_in_plus_in_minus) - \
                fxy(deltax_in_in * dx + dx / 2., deltay_in_in * dy - dy / 2.0, l2_in_minus_in_plus) + \
                fxy(deltax_in_in * dx + dx / 2., deltay_in_in * dy + dy / 2.0, l2_in_minus_in_minus)

    D_out_in_x = fxy(deltax_out_in * dx - dx / 2., deltay_out_in * dy - dy / 2.0, l2_out_plus_in_plus) - \
                 fxy(deltax_out_in * dx - dx / 2., deltay_out_in * dy + dy / 2.0, l2_out_plus_in_minus) - \
                 fxy(deltax_out_in * dx + dx / 2., deltay_out_in * dy - dy / 2.0, l2_out_minus_in_plus) + \
                 fxy(deltax_out_in * dx + dx / 2., deltay_out_in * dy + dy / 2.0, l2_out_minus_in_minus)

    # u_y measurements

    A_in_in_y = fxx(deltay_in_in * dy - dy / 2.0, deltax_in_in * dx - dx / 2., l2_in_plus_in_plus) - \
                fxx(deltay_in_in * dy + dy / 2.0, deltax_in_in * dx - dx / 2., l2_in_plus_in_minus) - \
                fxx(deltay_in_in * dy - dy / 2.0, deltax_in_in * dx + dx / 2., l2_in_minus_in_plus) + \
                fxx(deltay_in_in * dy + dy / 2.0, deltax_in_in * dx + dx / 2., l2_in_minus_in_minus)

    A_out_in_y = fxx(deltay_out_in * dy - dy / 2.0, deltax_out_in * dx - dx / 2., l2_out_plus_in_plus) - \
                 fxx(deltay_out_in * dy + dy / 2.0, deltax_out_in * dx - dx / 2., l2_out_plus_in_minus) - \
                 fxx(deltay_out_in * dy - dy / 2.0, deltax_out_in * dx + dx / 2., l2_out_minus_in_plus) + \
                 fxx(deltay_out_in * dy + dy / 2.0, deltax_out_in * dx + dx / 2., l2_out_minus_in_minus)

    D_in_in_y = fxy(deltay_in_in * dy - dy / 2.0, deltax_in_in * dx - dx / 2., l2_in_plus_in_plus) - \
                fxy(deltay_in_in * dy + dy / 2.0, deltax_in_in * dx - dx / 2., l2_in_plus_in_minus) - \
                fxy(deltay_in_in * dy - dy / 2.0, deltax_in_in * dx + dx / 2., l2_in_minus_in_plus) + \
                fxy(deltay_in_in * dy + dy / 2.0, deltax_in_in * dx + dx / 2., l2_in_minus_in_minus)

    D_out_in_y = fxy(deltay_out_in * dy - dy / 2.0, deltax_out_in * dx - dx / 2., l2_out_plus_in_plus) - \
                 fxy(deltay_out_in * dy + dy / 2.0, deltax_out_in * dx - dx / 2., l2_out_plus_in_minus) - \
                 fxy(deltay_out_in * dy - dy / 2.0, deltax_out_in * dx + dx / 2., l2_out_minus_in_plus) + \
                 fxy(deltay_out_in * dy + dy / 2.0, deltax_out_in * dx + dx / 2., l2_out_minus_in_minus)

    if not loworder:

        B_in_in_x = x_in[..., np.newaxis] * A_in_in_x - fxxx(deltax_in_in - dx / 2., deltay_in_in - dy / 2.0,
                                                             l2_in_plus_in_plus) + \
                    fxxx(deltax_in_in - dx / 2., deltay_in_in + dy / 2.0, l2_in_plus_in_minus) + \
                    fxxx(deltax_in_in + dx / 2., deltay_in_in - dy / 2.0, l2_in_minus_in_plus) - \
                    fxxx(deltax_in_in + dx / 2., deltay_in_in + dy / 2.0, l2_in_minus_in_minus)

        B_out_in_x = x_out[..., np.newaxis] * A_out_in_x - fxxx(deltax_out_in - dx / 2., deltay_out_in - dy / 2.0,
                                                                l2_out_plus_in_plus) + \
                     fxxx(deltax_out_in - dx / 2., deltay_out_in + dy / 2.0, l2_out_plus_in_minus) + \
                     fxxx(deltax_out_in + dx / 2., deltay_out_in - dy / 2.0, l2_out_minus_in_plus) - \
                     fxxx(deltax_out_in + dx / 2., deltay_out_in + dy / 2.0, l2_out_minus_in_minus)

        C_in_in_x = y_in[..., np.newaxis] * A_in_in_x - fxxy(deltax_in_in - dx / 2., deltay_in_in - dy / 2.0,
                                                             l2_in_plus_in_plus) + \
                    fxxy(deltax_in_in - dx / 2., deltay_in_in + dy / 2.0, l2_in_plus_in_minus) + \
                    fxxy(deltax_in_in + dx / 2., deltay_in_in - dy / 2.0, l2_in_minus_in_plus) - \
                    fxxy(deltax_in_in + dx / 2., deltay_in_in + dy / 2.0, l2_in_minus_in_minus)

        C_out_in_x = y_out[..., np.newaxis] * A_out_in_x - fxxy(deltax_out_in - dx / 2., deltay_out_in - dy / 2.0,
                                                                l2_out_plus_in_plus) + \
                     fxxy(deltax_out_in - dx / 2., deltay_out_in + dy / 2.0, l2_out_plus_in_minus) + \
                     fxxy(deltax_out_in + dx / 2., deltay_out_in - dy / 2.0, l2_out_minus_in_plus) - \
                     fxxy(deltax_out_in + dx / 2., deltay_out_in + dy / 2.0, l2_out_minus_in_minus)

        E_in_in_x = x_in[..., np.newaxis] * D_in_in_x - fxyx(deltax_in_in - dx / 2., deltay_in_in - dy / 2.0,
                                                             l2_in_plus_in_plus) + \
                    fxyx(deltax_in_in - dx / 2., deltay_in_in + dy / 2.0, l2_in_plus_in_minus) + \
                    fxyx(deltax_in_in + dx / 2., deltay_in_in - dy / 2.0, l2_in_minus_in_plus) - \
                    fxyx(deltax_in_in + dx / 2., deltay_in_in + dy / 2.0, l2_in_minus_in_minus)

        E_out_in_x = x_out[..., np.newaxis] * D_out_in_x - fxyx(deltax_out_in - dx / 2., deltay_out_in - dy / 2.0,
                                                                l2_out_plus_in_plus) + \
                     fxyx(deltax_out_in - dx / 2., deltay_out_in + dy / 2.0, l2_out_plus_in_minus) + \
                     fxyx(deltax_out_in + dx / 2., deltay_out_in - dy / 2.0, l2_out_minus_in_plus) - \
                     fxyx(deltax_out_in + dx / 2., deltay_out_in + dy / 2.0, l2_out_minus_in_minus)

        F_in_in_x = y_in[..., np.newaxis] * D_in_in_x - fxyx(deltax_in_in - dx / 2., deltay_in_in - dy / 2.0,
                                                             l2_in_plus_in_plus) + \
                    fxyx(deltax_in_in - dx / 2., deltay_in_in + dy / 2.0, l2_in_plus_in_minus) + \
                    fxyx(deltax_in_in + dx / 2., deltay_in_in - dy / 2.0, l2_in_minus_in_plus) - \
                    fxyx(deltax_in_in + dx / 2., deltay_in_in + dy / 2.0, l2_in_minus_in_minus)

        F_out_in_x = y_out[..., np.newaxis] * D_out_in_x - fxyx(deltax_out_in - dx / 2., deltay_out_in - dy / 2.0,
                                                                l2_out_plus_in_plus) + \
                     fxyx(deltax_out_in - dx / 2., deltay_out_in + dy / 2.0, l2_out_plus_in_minus) + \
                     fxyx(deltax_out_in + dx / 2., deltay_out_in - dy / 2.0, l2_out_minus_in_plus) - \
                     fxyx(deltax_out_in + dx / 2., deltay_out_in + dy / 2.0, l2_out_minus_in_minus)


        B_in_in_y = y_in[..., np.newaxis] * A_in_in_y - fxxx(deltay_in_in - dy / 2.0, deltax_in_in - dx / 2.,
                                                             l2_in_plus_in_plus) + \
                    fxxx(deltay_in_in + dy / 2.0, deltax_in_in - dx / 2., l2_in_plus_in_minus) + \
                    fxxx(deltay_in_in - dy / 2.0, deltax_in_in + dx / 2., l2_in_minus_in_plus) - \
                    fxxx(deltay_in_in + dy / 2.0, deltax_in_in + dx / 2., l2_in_minus_in_minus)

        B_out_in_y = y_out[..., np.newaxis] * A_out_in_y - fxxx(deltay_out_in - dy / 2.0, deltax_out_in - dx / 2.,
                                                                l2_out_plus_in_plus) + \
                     fxxx(deltay_out_in + dy / 2.0, deltax_out_in - dx / 2., l2_out_plus_in_minus) + \
                     fxxx(deltay_out_in - dy / 2.0, deltax_out_in + dx / 2., l2_out_minus_in_plus) - \
                     fxxx(deltay_out_in + dy / 2.0, deltax_out_in + dx / 2., l2_out_minus_in_minus)

        C_in_in_y = x_in[..., np.newaxis] * A_in_in_y - fxxy(deltay_in_in - dy / 2.0, deltax_in_in - dx / 2.,
                                                             l2_in_plus_in_plus) + \
                    fxxy(deltay_in_in + dy / 2.0, deltax_in_in - dx / 2., l2_in_plus_in_minus) + \
                    fxxy(deltay_in_in - dy / 2.0, deltax_in_in + dx / 2., l2_in_minus_in_plus) - \
                    fxxy(deltay_in_in + dy / 2.0, deltax_in_in + dx / 2., l2_in_minus_in_minus)

        C_out_in_y = x_out[..., np.newaxis] * A_out_in_y - fxxy(deltay_out_in - dy / 2.0, deltax_out_in - dx / 2.,
                                                                l2_out_plus_in_plus) + \
                     fxxy(deltay_out_in + dy / 2.0, deltax_out_in - dx / 2., l2_out_plus_in_minus) + \
                     fxxy(deltay_out_in - dy / 2.0, deltax_out_in + dx / 2., l2_out_minus_in_plus) - \
                     fxxy(deltay_out_in + dy / 2.0, deltax_out_in + dx / 2., l2_out_minus_in_minus)

        E_in_in_y = y_in[..., np.newaxis] * D_in_in_y - fxyx(deltay_in_in - dy / 2.0, deltax_in_in - dx / 2.,
                                                             l2_in_plus_in_plus) + \
                    fxyx(deltay_in_in + dy / 2.0, deltax_in_in - dx / 2., l2_in_plus_in_minus) + \
                    fxyx(deltay_in_in - dy / 2.0, deltax_in_in + dx / 2., l2_in_minus_in_plus) - \
                    fxyx(deltay_in_in + dy / 2.0, deltax_in_in + dx / 2., l2_in_minus_in_minus)

        E_out_in_y = y_out[..., np.newaxis] * D_out_in_y - fxyx(deltay_out_in - dy / 2.0, deltax_out_in - dx / 2.,
                                                                l2_out_plus_in_plus) + \
                     fxyx(deltay_out_in + dy / 2.0, deltax_out_in - dx / 2., l2_out_plus_in_minus) + \
                     fxyx(deltay_out_in - dy / 2.0, deltax_out_in + dx / 2., l2_out_minus_in_plus) - \
                     fxyx(deltay_out_in + dy / 2.0, deltax_out_in + dx / 2., l2_out_minus_in_minus)

        F_in_in_y = x_in[..., np.newaxis] * D_in_in_y - fxyx(deltay_in_in - dy / 2.0, deltax_in_in - dx / 2.,
                                                             l2_in_plus_in_plus) + \
                    fxyx(deltay_in_in + dy / 2.0, deltax_in_in - dx / 2., l2_in_plus_in_minus) + \
                    fxyx(deltay_in_in - dy / 2.0, deltax_in_in + dx / 2., l2_in_minus_in_plus) - \
                    fxyx(deltay_in_in + dy / 2.0, deltax_in_in + dx / 2., l2_in_minus_in_minus)

        F_out_in_y = x_out[..., np.newaxis] * D_out_in_y - fxyx(deltay_out_in - dy / 2.0, deltax_out_in - dx / 2.,
                                                                l2_out_plus_in_plus) + \
                     fxyx(deltay_out_in + dy / 2.0, deltax_out_in - dx / 2., l2_out_plus_in_minus) + \
                     fxyx(deltay_out_in - dy / 2.0, deltax_out_in + dx / 2., l2_out_minus_in_plus) - \
                     fxyx(deltay_out_in + dy / 2.0, deltax_out_in + dx / 2., l2_out_minus_in_minus)

        G_in_in_xx = A_in_in_x + B_in_in_x + C_in_in_x
        G_in_in_xy = D_in_in_x + E_in_in_x + F_in_in_x
        G_out_in_xx = (A_out_in_x + B_out_in_x + C_out_in_x)
        G_out_in_xy = (D_out_in_x + E_out_in_x + F_out_in_x)

        G_in_in_yy = A_in_in_y + B_in_in_y + C_in_in_y
        G_in_in_yx = D_in_in_y + E_in_in_y + F_in_in_y
        G_out_in_yy = (A_out_in_y + B_out_in_y + C_out_in_y)
        G_out_in_yx = (D_out_in_y + E_out_in_y + F_out_in_y)

    else:
        G_in_in_xx = A_in_in_x
        G_in_in_xy = D_in_in_x
        G_out_in_xx = A_out_in_x
        G_out_in_xy = D_out_in_x

        G_in_in_yy = A_in_in_y
        G_in_in_yx = D_in_in_y
        G_out_in_yy = A_out_in_y
        G_out_in_yx = D_out_in_y

    Dx = sparse.csr_matrix(
        (deltax_in_in == 0) * (deltay_in_in == 0) * -1 + (deltax_in_in == 1) * (deltay_in_in == 0) * 1)
    rowsums = np.squeeze(np.asarray((Dx.sum(axis=1) != 0)))
    Dx[rowsums, :] = 0
    Dx.eliminate_zeros()
    Dx = Constant(Dx)

    Dy = sparse.csr_matrix(
        (deltay_in_in == 0) * (deltax_in_in == 0) * -1 + (deltay_in_in == 1) * (deltax_in_in == 0) * 1)
    rowsums = np.squeeze(np.asarray((Dy.sum(axis=1) != 0)))
    Dy[rowsums, :] = 0
    Dy.eliminate_zeros()
    Dy = Constant(Dy)

    return G_in_in_xx, G_in_in_xy, G_out_in_xx, G_out_in_xy, G_in_in_yy, G_in_in_yx, G_out_in_yy, G_out_in_yx, Dx, Dy
