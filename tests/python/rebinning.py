#!/usr/bin/env python3
# encoding: utf-8

from __future__ import print_function
from scipy.linalg import block_diag
import numpy as np
import timeit

def make_rebin_matrix_1d(sizein, ntogroup):
    sizeout = sizein//ntogroup
    assert sizein%ntogroup ==0

    Kblock = np.ones((ntogroup,1), dtype='i')
    # print(f'Kblock: {Kblock}')
    K = block_diag(*[Kblock]*sizeout)

    return K

def make_rebin_matrices_2d(shapein, ntogroup):
    Kleft  = make_rebin_matrix_1d(shapein[0], ntogroup[0]).T
    Kright = make_rebin_matrix_1d(shapein[1], ntogroup[1])

    return Kleft, Kright

def check1d():
    sizein=12
    ngroup=4

    K = make_rebin_matrix_1d(sizein, ngroup)
    print(timeit.timeit(lambda: make_rebin_matrix_1d(sizein, ngroup), number=100))
    print(f'Size in: {sizein}')
    print(f'N to group: {ngroup}')
    print(f'Shape K: {K.shape}')
    print(f'K: {K!s}')
    print()

    arrin1  = np.ones((1,sizein), dtype='d')
    print(f'Shape1 in: {arrin1.shape}')

    arrout1 = np.matmul(arrin1, K)
    print(f'Shape1 out: {arrout1.shape}')
    print(f'Arr1 in: {arrin1!s}')
    print(f'Arr1 out: {arrout1!s}')
    print()

    arrin2  = np.ones((3,sizein), dtype='d')
    arrin2[1]*=2
    arrin2[2]*=3
    print(f'Shape2 in: {arrin2.shape}')

    arrout2 = np.matmul(arrin2, K)
    print(f'Shape2 out: {arrout2.shape}')
    print(f'Arr2 in: {arrin2!s}')
    print(f'Arr2 out: {arrout2!s}')
    print()

def check2d():
    shapein=(6, 4)
    ngroup=(3, 2)

    Kleft, Kright = make_rebin_matrices_2d(shapein, ngroup)
    print(timeit.timeit(lambda: make_rebin_matrices_2d(shapein, ngroup), number=100))

    print(f'N to group: {ngroup}')
    print(f'Shape K left: {Kleft.shape}')
    print(f'Shape in: {shapein}')
    print(f'Shape K right: {Kright.shape}')
    print(f'K left: {Kleft}')
    print(f'K right: {Kright}')
    print()

    arrin1  = np.arange(np.prod(shapein), dtype='d').reshape(shapein)
    print(f'Shape1 in: {arrin1.shape}')

    arrout1 = np.matmul(np.matmul(Kleft, arrin1), Kright)
    print(f'Shape1 out: {arrout1.shape}')
    print(f'Arr1 in: \n{arrin1!s}')
    print(f'Arr1 out: \n{arrout1!s}')
    print()

    arrin2 = np.ones(shapein, dtype='d')
    arrin2[:3, :2] = 1.0
    arrin2[:3, 2:] = 2.0
    arrin2[3:, :2] = 3.0
    arrin2[3:, 2:] = 4.0
    print(f'Shape2 in: {arrin2.shape}')

    arrout2 = np.matmul(np.matmul(Kleft, arrin2), Kright)
    print(f'Shape2 out: {arrout2.shape}')
    print(f'Arr2 in: \n{arrin2!s}')
    print(f'Arr2 out: \n{arrout2!s}')
    print()

def main(args):
    check1d()
    check2d()

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # parser.add_argument( 'input' )

    main( parser.parse_args() )
