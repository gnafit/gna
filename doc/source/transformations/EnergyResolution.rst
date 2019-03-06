.. _EnergyResolution:

EnergyResolution
~~~~~~~~~~~~~~~~

Description
^^^^^^^^^^^
Applies energy smearing matrix to the input histograms of events binned in :math:`E_{\text{vis}}`. Each bin of the
histogram is smeared according to Gaussian function with width, defined as a function of bin center.

.. The transformation may be configured within :ref:`detector_eres_common3` bundle.

Transformations
^^^^^^^^^^^^^^^

The object contains at least two transformations. One `matrix` transformation and at least one `smear` transformation.

Matrix transformation
"""""""""""""""""""""

The `matrix` transformation computes the smearing matrix depending on three parameters :math:`a`, :math:`b` and :math:`c`.

Inputs
''''''

1. ``matrix.Edges`` -- a histogram defining the bin edges. 

The transformations reads only the bins definitions. The actual data from the histogram is not requested and the
taintflag is not propagated.

Outputs
'''''''

1. ``matrix.FakeMatrix`` -- smearing matrix. 

By default the output matrix is used to propagate the taintstatus only. The actual matrix is sparse and is stored
internally. `propagate_matrix` option should be set in order to `matrix.FakeMatrix` to be written.

Variables
'''''''''

1. ``Eres_a`` — :math:`a`,
2. ``Eres_b`` — :math:`b` and
3. ``Eres_c`` — :math:`c`

are the parameters of the energy resolution formula. See below.

Smear transformation
""""""""""""""""""""

Inputs
''''''''''

#. ``smear.FakeMatrix`` — smearing matrix. Binded automatically when transformation is created.
#. ``smear.Ntrue`` — one-dimensional histogram of number of events :math:`N_{\text{vis}}`.
#. ``smear.Ntrue_02`` —  another histogram  of the same shape(optional).
#. ...

Outputs
''''''''''

#. ``smear.Nrec`` — one-dimensional smeared histogram of number of events :math:`N_{\text{rec}}`
#. ``smear.Nrec_02`` — the second histogram smeared.
#. ...

Subsequent transformations
''''''''''''''''''''''''''

Subsequent transformations are named `smear_02`, `smear_03` etc.

Arguments and functions
"""""""""""""""""""""""

There is a pythonic constructor defined in `gna.constructors`:

`EnergyResolution(parameters, propagate_matrix=False)`
    initializes energy resolution with a list of parameters (should contain three items) and a boolean flag that may
    trigger the `matrix.FakeMatrix` output to be written.

    The constructor creates a single `smear` transformation with a single input and corresponding output.

The `EnergyResolution` object contains the following methods:

`add_transformation(name='')`
    adds a new `smear` transformation with name, passed as input, or generated automatically. Returns new transformation.

`add_input(inputname='', output='name')`
    adds a new input/output pair to the last `smear` transformation. The names are either set as arguments or generated
    automatically. Returns newly created input.

`add_input(output, inputname='', output='name')`
    adds a new input/output pair to the last `smear` transformation. The names are either set as arguments or generated
    automatically. The output, passed as argument is connected to the newly created input. Returns new output.

The inputs of a single transformation are all processed, even if only one of them is tainted. For the description see
the :ref:`tutorial <tutorial_topology>`.

Also, since the constructor of the object creates a transformation with an input, first call to any of the `add_input()`
method will use this input if it is unbound.

Tests
^^^^^

Use the following commands for the usage example and testing:

.. code:: bash

   ./tests/detector/test_eres.py -s

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
    V^{\text{res}}_{ij} = \frac{1}{\sqrt{2 \pi} \sigma(E_j)} \exp \left( - \frac{(E_j - E_i)^2}{2 \sigma^2(E_j)} \right),

where :math:`\sigma(E_j)` is:

.. math::
    \sigma(E_j) = E_j \sqrt{ a^2 + \frac{b^2}{E_j}  + \left( \frac{c}{E_j}\right)^2}

where :math:`a`, :math:`b`, :math:`c` are resolution parameters.

.. figure:: ../../img/eres_scheme.png
   :scale: 25 %
   :align: center

   Energy resolution bundle scheme.



