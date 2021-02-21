# from optspace cimport OptSpace as OptSpace_c
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

cdef extern from "OptSpace_C/OptSpace.h":
    # sparse matrix structure.
    cdef struct smat:
        long rows
        long cols
        long vals     # Total non-zero entries.
        long *pointr  # For each col (plus 1), index of first non-zero entry.
        long *rowind  # For each nz entry, the row index.
        double *value # For each nz entry, the value.

    ctypedef smat* SMat

    # reconstruction results.
    cdef struct reconvar:
        int rows;
        int cols;
        int rank;
        # * * * * The "left" subspace * * * *
        double **X;
        #* * * * The "right" subspace * * * *
        double **Y;
        #* * * * Argmin of f(X,Y,S) * * * *
        double **S;

    ctypedef reconvar* ReconVar

    ReconVar OptSpace(SMat M,int r,int niter,double tol,int Verbosity)#,char *outfile);

cdef extern from "OptSpace_C/svdlib.h":
    SMat svdNewSMat(int rows, int cols, int vals);

import numpy as np
import random

def optspace(sparse_mat, rank_n, num_iter, tol, verbosity, outfile):
    '''
    compute rank-n matrix factorization of the sparse_mat

    Params
    =====
    sparse_mat (dict): (i,j)->val, a sprase matrix.
    rows: number of rows in the sparse matrix.
    cols: number of columns in the sparse matrix.
    rank_n: the rank-n parameter.
    num_iter: number of iterations.
    tol: algorithm stops if relative error ||P_E(XSY'-M)||_F/||P_E(M)||_F < tol
    verbosity:  0 (quiet), 1 (print rmse and rel. error) , 2
    outfile: filename for writing (by iteration) rmse and rel. error ("" to disable)

    Returns
    =======
    '''
    # convert input into c format.
    rows = max([i+1 for (i,j,_) in sparse_mat])
    cols = max([j+1 for (i,j,_) in sparse_mat])

    # !IMPORTANT: has to be in this order to have a big gradient.
    sparse_mat = sorted(sparse_mat, key=lambda tp: tp[1] * rows + tp[0])

    vals = len(sparse_mat)

    cdef int *degree_column = <int*> PyMem_Malloc(cols * sizeof(int))
    for i in range(cols):
        degree_column[i] = 0

    M = svdNewSMat(rows, cols, (int)(1.1 * float(vals)))

    cdef int _count = 0;
    for (i, j, val) in sparse_mat:
        M.rowind[_count] = i
        M.value[_count]  = val
        degree_column[j] += 1
        _count += 1

    M.rows = rows
    M.cols = cols
    M.vals = vals

    M.pointr[0] = 0

    for i in range(1, cols + 1):
        M.pointr[i] = M.pointr[i-1] + degree_column[i-1]
    # apply optspace algorithm.





    #print(outfile)
    recon_var = OptSpace(M, rank_n, num_iter, tol, verbosity)

    # convert result to numpy format.
    X_np = np.zeros((rows, rank_n))
    for i in range(rows):
        for j in range(rank_n):
            X_np[i][j] = recon_var.X[i][j]

    Y_np = np.zeros((cols, rank_n))
    for i in range(cols):
        for j in range(rank_n):
            Y_np[i][j] = recon_var.Y[i][j]

    S_np = np.zeros((rank_n, rank_n))
    for i in range(rank_n):
        for j in range(rank_n):
            S_np[i][j] = recon_var.S[i][j]

    return (X_np, S_np, Y_np)

