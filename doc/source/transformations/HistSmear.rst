HistSmear
~~~~~~~~~

Description
^^^^^^^^^^^
Applies (energy) smearing matrix to the histogram of events binned in :math:`E_{\text{vis}}`.

HistSmear_ class may be used to implement, for example, Daya Bay IAV smearing.
The IAV uncertainty may be implemented by connecting the output of RenormalizeDiag transformation
as the input to HistSmear_.

Inputs
^^^^^^

1. ``'smear.Ntrue'`` — one-dimensional histogram of number of events :math:`N_{\text{true}}`.
2. ``'smear.SmearMatrix'`` —­square smearing matrix of number of events :math:`C`.

Outputs
^^^^^^^

1. ``'smear.Nvis'`` — one-dimensional smeared histo of number of events :math:`N_{\text{vis}}`

Arguments
^^^^^^^^^

1. ``bool upper``. If true HistSmear_ will ensure that the matrix is upper diagonal.
   Useful for the cases of energy leak type smearing.

Tests
^^^^^

Use the following commands for the usage example and testing:

.. code:: bash

   ./tests/detector/test_esmear.py -s
   ./tests/detector/test_iavunc.py -s

Implementation
^^^^^^^^^^^^^^

The smeared histo :math:`N_{\text{vis}}` and true :math:`N_{\text{true}}` are connected through a matrix transformation:

.. math::
   N^{\text{rec}}_i = \sum_j C_{ij} N^{\text{vis}}_j,

where :math:`N^{\text{rec}}_i` is a reconstructed number of events in a *i*-th
bin, :math:`N^{\text{vis}}_j` is a true number of events in a *j*-th bin and
:math:`C_{ij}` is a probability for events to flow from *j*-th to
*i* bin.

The matrix :math:`C` usually satisfies the following condition:

.. math::
   \sum_i C_{ij} = 1

for most of the histogram bins. Exceptions are bins in the beginning and end of the histogram.
Events from these bins leak outside the histogram, so the sum may be less than one.

