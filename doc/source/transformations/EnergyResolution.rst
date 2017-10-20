EnergyResolution
~~~~~~~~~~~~~~~~

Description
^^^^^^^^^^^
Applies energy resolution to the histogram of events binned in :math:`E_{\text{vis}}`.

Inputs
^^^^^^

1. ``'smear.Nvis'`` — one-dimensional histogram of number of events :math:`N_{\text{vis}}`.

Outputs
^^^^^^^

1. ``'smear.Nrec'`` one-dimensional smeared histo of number of events :math:`N_{\text{rec}}`

Variables
^^^^^^^^^

1. ``Eres_a`` — :math:`a`,
1. ``Eres_b`` — :math:`b`,
1. ``Eres_c`` — :math:`c`

are the parameters of the energy resolution formula. See below.

Implementation
^^^^^^^^^^^^^^

The smeared histo :math:`N_{\text{rec}}` and true :math:`N_{\text{vis}}` are connected through a matrix transformation:

.. math::
   N^{\text{rec}}_i = \sum_j V^{\text{res}}_{ij} N^{\text{vis}}_j,

where :math:`N^{\text{rec}}_i` is a reconstructed number of events in a *i*-th
bin, :math:`N^{\text{vis}}_j` is a true number of events in a *j*-th bin and
:math:`V^{\text{res}}_{ij}` is a probability for events to flow from *j*-th to
*i* bin.

That probability is given by:

.. math::
    V^{\text{res}}_{ij} = \frac{1}{\sqrt{2 \pi} \sigma(E_j)} \exp \left( - \frac{(E_j - E_i)^2}{2 \sigma(E_j)} \right),

where :math:`\sigma(E_j)` is:

.. math::
    \sigma(E_j) = E_j \sqrt{ a^2 + \frac{b^2}{E_j}  + \left( \frac{c}{E_j}\right)^2}

where :math:`a`, :math:`b`, :math:`c` are resolution parameters.
