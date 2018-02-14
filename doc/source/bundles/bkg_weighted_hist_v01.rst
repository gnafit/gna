.. _bkg_weighted_hist_v01:

Weighted background (version 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Overview
""""""""

The bundle ``bkg_weighted_hist_v01`` implements a histogram that is multiplied by a weight computed as a product of
several variables.

Scheme
""""""

For each variant:
  - Parse the formula:
    + initialize the variables if the values and uncertainties are provided within current configuration.
    + require the other variables.
    + create the target variable and define it as a product of input variables.
  - Build the chain:
    + For each variant create a :ref:`WeightedSum` instance that weights the input histogram with calculated weight.

Parameters
""""""""""

1. Parameters are defined according to the ``formula`` specification. See `Weight formula`_.

Outputs and observables
"""""""""""""""""""""""

The bundle provides the :ref:`WeightedSum` and its output for each variant.

.. code-block:: python

    self.transformations_out[ns.name] = ws.sum
    self.outputs[ns.name]             = ws.sum.sum

The observable (``ws.sum.sum``) is created in each namespace. The name is either read from configuration file or
determined based on a key from a parent dictionary (such that ``parent[key]==cfg``).

Configuration
"""""""""""""

Required parameters:
  - ``bundle`` -- the bundle name.
  - ``formula`` -- the histogram weight formula, that specifies the variable to store the weight and how this weight is
    computed. See below for examples.
  - ``groups`` (optional) -- groups and categories specification.
  - ``spectra`` -- configuration for the bundle, that will provide the spectrum. See :ref:`root_histograms_v01` for
    example.
  - ``variants`` -- list of strings used to create multiple namespaces in common_namespace. The bundle will provide an
    output for each namespace name.

Optional parameters:
  - ``name`` -- the background name. If omitted the background name is determined as a key from the parent dictionary.
    The name is required in case there is no parent.
  - Predefined input parameters, see explanation for the ``formula``.

Weight formula
""""""""""""""

The formula may be defined via one of the equivalent ways:
  - string, for example 'target=num1*num2'.
  - string pair, for example ('target', 'num1*num2').
  - string and list of strings ('target', ['num1', 'num2']).

``target`` is the name of the parameter that will store the histogram weight. ``num1``, ``num2`` are the parameter names
which are used within product to produce the target value.

Each of the names may contains periods '.' that is interpreted as nesting: ``'a.b'`` will create namespace ``'a'`` with
variable ``'b'``.

The name may also contain the formatting symbols which will be replaced according to the ``groups`` specification.

For example the formula ``'{det}.bkg2_num=bkg2_rate.{site}*{det}.livetime'`` when applied for the detector ``D1`` will
create or use namespace ``D1`` with variable ``bkg2_num``. The ``bkg2_num`` will be calculated as a product of variable
``G1`` from the namespace ``bkg2_rate`` (since ``D1`` is in group ``G1``) and variable ``livetime`` from the namespace
``D1``.

Such an approach is needed to group the minimization parameters in separate groups/namespaces. In this case the
``bkg2_rate`` parameter is known with uncertainty and thus is stored in a distinct namespace. ``livetime`` is supposed
to be constant and ``bkg2_num`` is calculated based on other variables, therefore they are stored in the same namespace.

The source variables used in the product either:
  - created before by other bundles or by a user.
  - created by the bkg_weighted_hist_v01 bundle.

The bundle will initialize the required variables by itself if it finds the relevant field in the configuration. For
example for item ``'bkg1_norm.{det}'`` the bundle will check ``bkg1_norm`` field to read the values and uncertainties.

Example configuration 1
"""""""""""""""""""""""

The following configuration defines a background for 4 detectors (variants). The 4 spectra are read from the root file.
The bundle initializes variables ``bkg1_norm`` and ``bkg1_rate`` for each detector. The variables ``livetime`` is
supposed to be initialized beforehand.

.. code-block:: python

    detectors = [ 'D1', 'D2', 'D3', 'D4' ]
    groups=NestedDict(
            exp  = { 'testexp': detectors },
            det  = { d: (d,) for d in detectors },
            site = NestedDict([
                ('G1', ['D1', 'D2']),
                ('G2', ['D3']),
                ('G3', ['D4'])
                ])
            )

    cfg = NestedDict(
            bundle   = 'bkg_weighted_hist_v01',
            formula  = [ '{det}.bkg1_num', ('bkg1_norm.{det}', '{det}.bkg1_rate', '{det}.livetime') ],
            groups   = groups,
            variants = detectors,

            bkg1_norm = uncertaindict([
                (det, (1.0, 1.0, 'percent')) \
                  for det in detectors
                ]),

            bkg1_rate = uncertaindict(
                  [ ('D1', 8),
                    ('D2', 7),
                    ('D3', 4),
                    ('D4', 3) ],
                    mode = 'fixed',
                ),

            spectra = NestedDict(
                bundle = 'root_histograms_v01',
                filename   = 'filename.root',
                format = 'hist_{}',
                variants = OrderedDict([
                    ( 'D1', 'G1_D1' ),
                    ( 'D2', 'G1_D2' ),
                    ( 'D3', 'G2_D3' ),
                    ( 'D4', 'G3_D4' ),
                    ]),
                normalize = True,
                )
            )

Example configuration 2
"""""""""""""""""""""""

The following configuration defines a background for 4 detectors (variants). The spectra are initialized for each site
(not detector) by the :ref:`dayabay_fastn_v01` bundle. The rate ``bkg_fn_rate`` is also defined for each site. The
livetime is supposed to be initialized beforehand.

.. code-block:: python

    detectors = [ 'D1', 'D2', 'D3', 'D4' ]
    groups=NestedDict(
            exp  = { 'testexp': detectors },
            det  = { d: (d,) for d in detectors },
            site = NestedDict([
                ('G1', ['D1', 'D2']),
                ('G2', ['D3']),
                ('G3', ['D4'])
                ])
            )

    cfg = NestedDict(
            bundle = 'bkg_weighted_hist_v01',
            formula = [ '{det}.bkg_fn_num', ('bkg_fn_rate.{site}', '{det}.livetime') ],
            groups = groups,
            variants = detectors,

            bkg_fn_rate = uncertaindict(
               [('G1', (1.0, 0.3)),
                ('G2', (3.0, 0.2)),
                ('G3', (2.0, 0.1))],
                mode = 'absolute',
                ),
            spectra = NestedDict(
                bundle='dayabay_fastn_v01',
                formula='fastn_shape.{site}',
                groups=groups,
                normalize=(0.7, 12.0),
                bins =N.linspace(0.0, 12.0, 241),
                order=2,
                pars=uncertaindict(
                   [ ('G1', (70.00, 0.1)),
                     ('G2', (60.00, 0.05)),
                     ('G3', (50.00, 0.2)) ],
                     mode='relative',
                    ),
                )
            )

Testing scripts
"""""""""""""""

The bundle ``bkg_weighted_hist_v01`` is tested within ``bkg_weighted_hist_v01`` testing script:

.. code-block:: sh

    ./tests/bundle/bkg_weighted_hist_v01.py


