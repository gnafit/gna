#!/usr/bin/env python

import numpy as np
import os
import pytest
from gna.unittest import allure_attach_file, savegraph
from gna import constructors as C

@pytest.mark.parametrize("dimension, seed", [(1, 1), (4, 4), (8, 8), (12, 12), (15, 15)])
def test_MatrixRelToAbs(dimension, seed, tmp_path):
    """Matrix transformation from relative to absolute"""
    MEAN_RANDOM_VALUE = 0.5
    MAX_SPECTRA_VALUE = 10
    MAX_MATRIX_VALUE = 10

    def generate_answer(matrix, spectra):
        spectra_len = len(spectra)
        product = np.tensordot(spectra, spectra.reshape(spectra_len, 1), axes=0)
        product = product.reshape((spectra_len, spectra_len))
        return product * matrix

    np.random.seed(seed)

    spectra = np.random.rand(dimension) * MAX_SPECTRA_VALUE
    matrix = np.random.rand(dimension, dimension)
    matrix = (matrix - MEAN_RANDOM_VALUE) / 2.0 * MAX_MATRIX_VALUE

    trans_spectra = C.Points(spectra, labels="Spectra")
    trans_matrix = C.Points(matrix, labels="Matrix")

    matrix_rel_to_abs = C.MatrixRelToAbs()
    matrix_rel_to_abs.product.spectra(trans_spectra)
    matrix_rel_to_abs.product.matrix(trans_matrix)

    result = matrix_rel_to_abs.product.product()
    check = generate_answer(matrix, spectra)
    diff = np.fabs(result - check)
    assert (result == check).all()

    path = os.path.join(str(tmp_path), f"graph_{dimension}.png")
    savegraph(matrix_rel_to_abs.product, path, verbose=False)
    allure_attach_file(path)
