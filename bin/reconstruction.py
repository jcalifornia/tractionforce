#!/usr/bin/env python3

import sys
import numpy as np
import cvxpy
from cvxpy import Variable, Minimize, sum_squares, norm, Problem, Parameter, mul_elemwise, sum_entries, Constant
from scipy import sparse
import sys, argparse
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tractionforce.elasticity import *
import gc
from tractionforce.norms import *
from tractionforce.problem import *

from cvxpy.atoms.elementwise.log import log as cvxlog
from cvxpy.atoms.elementwise.power import power as cvxpower

def read_data(filename):

    # read the boundary points and get a mask
    raw_data = np.loadtxt(filename,skiprows=0)
    coords = raw_data[:,:2]
    deflection = raw_data[:,2]
    boundary = raw_data[:,3]
    return  coords, deflection, boundary

def main():
    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)

    parser = MyParser()
    parser.add_argument("-f", "--file", dest="filename",
                        help="data file in CSV format", metavar="FILENAME")

    parser.add_argument("-o", "--output", dest="output",
                        help="output in CSV format", metavar="OUTPUT")

    parser.add_argument("-ofig", "--outfigures", dest="outfigures",
                        help="output plots in PDF format", metavar="OUTFIGURES")

    parser.add_argument("-t", "--threshold", dest="threshold",
                        help="threshold distance", metavar="THRESHOLD")

    parser.add_argument("-r", "--regularization", dest="regularization",
                        help = "tv, tviso, divergence, tvtrace", metavar="REGULARIZATION")

    parser.add_argument("-n","--nsolutions", dest="nsolutions",
                        help = "number of solutions", metavar = "NSOLUTIONS")

    parser.add_argument("-s", "--solver", dest="solver",
                        help="solver (cvxopt, ecos)", metavar="SOLVER")

    results = parser.parse_args()

    figure_outfile = "figureoutput.pdf" if results.outfigures is None else results.outfigures
    csv_outfile = "fittedout.csv" if results.output is None else results.output
    CUTOFF = 16 if results.threshold is None else float(results.threshold)
    REGULARIZATION = "tvnorm" if results.regularization is None else results.regularization
    N_SOLUTIONS = 10 if results.nsolutions is None else int(float(results.nsolutions))

    coords, deflection, boundary = read_data(results.filename)

    # let's see if we have gridded data. If we do, then use the implicit data grid for all computations
    x_obs_positions = sorted(set(coords[:, 0]))
    y_obs_positions = sorted(set(coords[:, 1]))

    dx = abs(x_obs_positions[1]-x_obs_positions[0])
    dy = abs(y_obs_positions[1]-y_obs_positions[0])

    N = len(x_obs_positions)
    M = len(y_obs_positions)

    boundary2d = boundary.reshape((N, M))
    mask = np.zeros(boundary2d.shape)
    for r in range(boundary2d.shape[1]):
        pts = np.where(boundary2d[:, r] == 1)
        if (len(pts[0]) > 0):
            mini = (min(min(pts)))
            maxi = max(max(pts))
            mask[ mini:maxi, r] = 1

    distances2d = -ndimage.distance_transform_edt(mask) + ndimage.distance_transform_edt(1 - mask)
    distances2d = distances2d.flatten()

    condition_inside = distances2d<=0
    condition_outside = (distances2d>0) * (distances2d<=CUTOFF)

    del distances2d, mask, boundary2d
    gc.collect()

    x_out = np.array(coords[condition_outside,0]/dx,dtype=int)
    y_out = np.array(coords[condition_outside,1]/dy,dtype=int)

    x_in = np.array(coords[condition_inside,0]/dx,dtype=int)
    y_in = np.array(coords[condition_inside,1]/dy,dtype=int)

    x_center = np.mean(x_in)
    y_center = np.mean(y_in)

    u_x_in = deflection[condition_inside]
    u_x_out = deflection[condition_outside]

    n_in = len(x_in)
    n_out = len(x_out)

    spacing = 1

    G_in_in_xx, G_in_in_xy, G_out_in_xx, G_out_in_xy, G_in_in_yy, G_in_in_yx, G_out_in_yy, G_out_in_yx, Dx, Dy = gen_matrices(
        x_in, y_in, x_out, y_out, dx * spacing, dy * spacing, loworder=True)

    print("Size of the problem is " + str( n_in + n_out))

    """
    Setting up the problem
    ======================
    We compute the coefficient matrices for the linear problem
    """






    """
    Setting up the optimization problem
    ===================================

    Define norms
    """

    gamma = Parameter(sign="positive",value=1)

    sigma_xz = Variable(n_in)
    sigma_yz = Variable(n_in)
    # predicted_in = A_in_in*sigma_xz + D_in_in*sigma_yz # add higher order terms
    predicted_in_x = G_in_in_xx * sigma_xz + G_in_in_xy * sigma_yz
    predicted_out_x = G_out_in_xx * sigma_xz + G_out_in_xy * sigma_yz
    predicted_in_y = G_in_in_yx * sigma_xz + G_in_in_yy * sigma_yz
    predicted_out_y = G_out_in_yx * sigma_xz + G_out_in_yy * sigma_yz
    gamma_vals = np.logspace(-3, 2, N_SOLUTIONS)

    error = sum_squares(u_x_in - predicted_in_x) + sum_squares(u_x_out - predicted_out_x)

    if REGULARIZATION == "tvtrace":
        regularity_penalty = tvnorm_trace_2d(sigma_xz,sigma_yz,Dx,Dy)
    elif REGULARIZATION == "tviso":
        regularity_penalty = norm(Dx * sigma_xz / dx, 1) + norm(Dy * sigma_xz / dy, 1) + norm(Dx * sigma_yz / dx, 1) + norm(
        Dy * sigma_yz / dy, 1)
    elif REGULARIZATION == "tv":
        regularity_penalty = tvnorm2d(sigma_xz, Dx, Dy) + tvnorm2d(sigma_yz, Dx, Dy)
    elif REGULARIZATION == 'l2_grad':
        regularity_penalty = sum_squares(Dx*sigma_xz + Dx*sigma_yz) + sum_squares(Dy*sigma_xz + Dy*sigma_yz)
    elif REGULARIZATION == 'l1':
        regularity_penalty = norm(sigma_xz+sigma_yz,p=1)
    elif REGULARIZATION == 'l2':
        regularity_penalty = sum_squares(sigma_xz + sigma_yz) + sum_squares(sigma_xz + sigma_yz)
    elif REGULARIZATION == 'det':
        gamma_vals = np.logspace(-8, -5, N_SOLUTIONS)
        regularity_penalty = sum_entries(-log2(sigma_xz)-log2(sigma_yz))
    else:
        print("Invalid regularization choice")
        sys.exit(0)

    forceconstraints = [sum_entries(sigma_xz)==0, sum_entries(sigma_yz)==0] # add torque-free constraint here
    net_torque = sum_entries(mul_elemwise(x_in-x_center,sigma_yz) - mul_elemwise(y_in-y_center,sigma_xz))

    torqueconstraints = [net_torque == 0]

    constraints = forceconstraints + torqueconstraints

    objective = Minimize(error + gamma*regularity_penalty)

    prob = Problem(objective, constraints)

    sq_penalty = []
    l1_penalty = []
    sigma_xz_values = []
    sigma_yz_values = []


    with PdfPages(figure_outfile) as pdf:
        for val in gamma_vals:
            gamma.value = val
            try:
                if results.solver is not None and results.solver == "ecos":
                    prob.solve(verbose= True, max_iters = 50,
                               warm_start=True, solver = cvxpy.ECOS,
                               feastol = 1e-6, reltol = 1e-5,
                               abstol = 1e-6)
                elif results.solver is not None and results.solver == "cvxopt":
                    prob.solve(verbose= True, max_iters = 50,
                               warm_start=True, solver = cvxpy.CVXOPT,
                               feastol = 1e-6, reltol = 1e-5,
                               abstol = 1e-6)
                else:
                    prob.solve(verbose=True, max_iters=50,
                               warm_start=True,
                               feastol=1e-6, reltol=1e-5,
                               abstol=1e-6)

            except cvxpy.SolverError:
                continue

            sq_penalty.append(error.value)
            l1_penalty.append(regularity_penalty.value)
            sigma_xz_values.append(sigma_xz.value)
            sigma_yz_values.append(sigma_yz.value)

            force = np.zeros_like(coords)
            force[condition_inside,0] = sigma_xz.value.reshape((n_in,))
            force[condition_inside,1] = sigma_yz.value.reshape((n_in,))

            u_x = np.zeros(coords.shape[0])
            u_x[condition_inside] = predicted_in_x.value
            u_x[condition_outside] = predicted_out_x.value

            maxmagnitude = np.max(np.abs(force))

            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            plt.figure(figsize=(10, 10))

            x_min = min(coords[boundary == 1, 0])
            x_max = max(coords[boundary == 1, 0])
            y_min = min(coords[boundary == 1, 1])
            y_max = max(coords[boundary == 1, 1])

            pdf.attach_note("$\gamma$: "+ str(val))

            plt.suptitle("$\gamma$: "+ str(val) + "\n" +
                         "mismatch: " + str(error.value) + " penalty: " + str(regularity_penalty.value))

            plt.subplot(221)
            plt.xlim((x_min - 40, x_max + 40))
            plt.ylim((y_min - 40, y_max + 40))
            plt.pcolormesh(x_obs_positions,y_obs_positions,force[:,0].reshape((len(x_obs_positions),len(y_obs_positions))).transpose(),
                           cmap='seismic_r',vmax = maxmagnitude*.75, vmin=-maxmagnitude*.8)
            plt.title("$\sigma_{xz}$")
            plt.colorbar()

            plt.subplot(222)
            plt.xlim((x_min - 40, x_max + 40))
            plt.ylim((y_min - 40, y_max + 40))
            plt.pcolormesh(x_obs_positions,y_obs_positions,force[:,1].reshape((len(x_obs_positions),len(y_obs_positions))).transpose(),
                           cmap='seismic_r',vmax = maxmagnitude*.75, vmin=-maxmagnitude*.8)
            plt.title("$\sigma_{yz}$")
            plt.colorbar()

            plt.subplot(223)
            plt.xlim((x_min - 40, x_max + 40))
            plt.ylim((y_min - 40, y_max + 40))
            plt.pcolormesh(x_obs_positions,y_obs_positions,u_x.reshape((len(x_obs_positions),len(y_obs_positions))).transpose(),
                           cmap='seismic_r')
            plt.title("$\hat{u}_x$")
            plt.colorbar()

            plt.subplot(224)
            plt.xlim((x_min - 40, x_max + 40))
            plt.ylim((y_min - 40, y_max + 40))
            plt.pcolormesh(x_obs_positions,y_obs_positions,(deflection - u_x).reshape((len(x_obs_positions),len(y_obs_positions))).transpose(),
                           cmap='seismic_r')
            plt.title("$u_x-\hat{u}_x$")
            plt.colorbar()
            pdf.savefig()
            #plt.show()
            plt.close()

        plt.plot( sq_penalty, l1_penalty)
        plt.xlabel("Mismatch", fontsize=16)
        plt.ylabel("Regularity", fontsize=16)
        plt.title('Trade-Off Curve', fontsize=16)

        l_curve_distances = np.abs((l1_penalty[-1]-l1_penalty[0])*sq_penalty -  \
            (sq_penalty[-1]-sq_penalty[0])*l1_penalty+sq_penalty[-1]*l1_penalty[0]-l1_penalty[-1]*sq_penalty[0])

        # Choose the optimal lambda value




        pdf.savefig()
        plt.close()


    input("Press Enter to continue...")

if __name__ == "__main__":
    main()