Reactor experiments
======================

The Gaussian peak example we have used before is a very simple
experiment model. Inclusion of some real experiments into analysis
requires implementation of the whole computational graphs
producing experimentally observable values. The graph
construction may be designed to be done partially, with separate
modules defining or extending its parts. In the end you'll again have
one (or more) observables, which will be later employed to comparison
with experimental data, and a number of parameters to be fitted or
fixed.

At the moment, the only implemented experimental model which have
physical sense is the reactor experiment model. It's quite simple and
straightforward though and does only the primitive event number
computation.

The computation of observable spectrum is generally done with the
following formula:

.. math::
   N_i = \int\limits_{E_i}^{E_{i+1}} d E
   \frac{dE_{\bar{\nu}}}{dE}\sigma(E_{\bar{\nu}}) P_{ee}(E_{\bar{\nu}}) \sum \limits_k
   n_k S_k(E_{\bar{\nu}})

:math:`N_i` is the event number in the :math:`i`-th bin containing
events with energies :math:`[E_i, E_{i+1}]`; :math:`E` and
:math:`E_{\bar{\nu}}` are positron and neutrino energies;
:math:`\sigma(E_{\bar{\nu}})` is the IBD cross section; :math:`P_{ee}(E_{\bar{\nu}})` is
the oscillation probability; :math:`S_k(E_{\bar{\nu}})` is the antineutrino
spectrum of the :math:`k`-th isotope with corresponding normalization
:math:`n_k`.

The general vacuum oscillation probability can be represented in the following form:

.. math::
   P(E_{\bar{\nu}}) = \sum_j w_j(\theta) P_j(\Delta m^2, E_{\bar{\nu}})

where the mixing angle only dependent weights :math:`w_j` and mass
difference dependent oscillatory terms :math:`P_j` are
factorized. Since only the oscillatory terms are energy dependent,
when doing fits it's more computationally efficient to take the
weights out of the integrals, making the recomputations a lot faster
if only mixing angles are changed.

The :math:`P_j(E_{\bar{\nu}})` functions and other derived :math:`j`-indexed
functions are generally called *components*.

The corresponding formula for :math:`P_{ee}(E_{\bar{\nu}})` is:

.. math::
    P_{ee}(E_{\bar{\nu}}) = 1 - \cos^4 \theta_{13} \sin^2 2\theta_{12} \sin^2 \Delta_{21}
                            - \sin^2 2\theta_{13} \cos^2 \theta_{12} \sin^2 \Delta_{31}
                            - \sin^2 2\theta_{13} \sin^2 \theta_{12} \sin^2 \Delta_{32}

where :math:`\Delta_{ij} = \dfrac{\Delta m^2_{ij} L}{4E}` and you can see what
are :math:`w_j` and :math:`P_j`.

The corresponding expression for number of events in a given bin, which is
implemented in code reads:

.. math::
   N_i = \sum_j w_j \int\limits_{E_i}^{E_{i+1}} d E
   \frac{dE_{\bar{\nu}}}{dE}\sigma(E_{\bar{\nu}}) P_j(E_{\bar{\nu}}) \sum \limits_k
   n_k S_k(E_{\bar{\nu}})


The given formula are only for the case of one reactor. If there are
several of them, an additional summation inside the integral should be
performed.

All of that is implemented in the
`gna.exp.reactor.ReactorExperimentModel` class. As the most basic
usage you should just specify the properties of each detector and
reactor of the experiment. The class checks the currently
available namespaces for required parameters and creates missing parameters and
set up computational graph. It is not a complete list of namespaces that will be created, *always check the code* for the details: 

- ``ns.ibd``: namespace for parameters related to the IBD cross sections, they are defined
  in ``gna.parameters.ibd`` and are usually fixed;
- ``ns.oscillation``: namespace for oscillations parameters (mass splittings,
  squared sines of mixing angles, hierarchy (``Alpha`` --
  ``normal`` or numerically 1 for normal, ``inverted`` or -1 for
  inverted), as defined in ``gna.parameters.oscillation``, :math:`\delta_{\text{CP}}` ;
- ``ns.detectors.detname``: namespace for parameters related to detector
  ``detname``, namely: ``TargetProtons`` for protons number (check
  ``neutrino/ReactorNorm.cc``); ``Eres_a``, ``Eres_b``, ``Eres_c`` for
  the corresponding parameters of the resolution formula (check ``detector/EnergyResolution.cc``) ;
- ``ns.reactors.rname``: parameters related to one reactor:
  ``ThermalPower`` for the nominal thermal power in GW,
  used in ``ReactorNorm``;
- ``ns.detname.rname``: parameters related to detector/reactor pair,
  for example ``L`` is the distance between reactor and detector in
  km.

Here ``ns`` stands for namespace of the experiment.

Let's see how this works on example of the JUNO experiment. The
initialization is done in the ``pylib/gna/ui/juno.py``:

.. caution:: 

    This section is outdated. The file is no longer supported.

In the ``initparser()`` classmethod we just call the corresponding
methods in ``basecmd`` and ``ReactorExperimentModel`` to define the
arguments typical to a reactor experiment. In the ``init()`` we
actually instantiate the ``ReactorExperimentModel``, providing just
lists of reactors and detectors. All the results (``observables``) are
stored internally in the ``env``, so the resulting object is not saved
explicitly, but it of course can be done if needed.

Each reactor and detector is represented with a ``Reactor`` or ``Detector``
object. They just hold configuration, that should be provided at
instantiation through kwargs. The following options are available for
``Reactor``:

- ``name`` -- unique string which will identify the reactor in
  namespace, should not contain spaces and must not contain dots;
- ``location`` -- coordinates, an ``np.array()``-able numeric
  expected. It will be subtracted with the detectors locations and
  Euclidian norm will be taken later. A number or 3-dimensonal
  coordinates will work, but choose should be consistent with the ones of the
  Detectors.
- ``power`` -- nominal thermal power of the reactor, in GW;
- ``power_rate`` -- an array with actual power splitted by periods;
  splitting is arbitrary but should be consistent across all the
  reactors and detectors;
- ``fission_fractions`` -- a dict mapping from strings (isotope names)
  to array of fission rates by period (the same splitting as previous
  one).

``Detector`` has the following options:

- ``name`` -- unique string identifier, please no spaces and
  definetely no dots or slashes;
- ``location`` -- coordinaties in the same format as in the
  ``Reactor``;
- ``edges`` -- edges of the default binning, in MeV of visible energy;
- ``protons`` -- number of protons of detector target;
- ``livetime`` -- length of the periods in seconds.

As you can see, we have one period of 5 years and since we have only
one detector, we can use one-dimensional coordinates for
simplicity. Totally 12 reactors are defined and one detector with name
``AD1``. To speed up the calculations, reactors with similar distance
to the detector are grouped into one *reactor group* by approximation,
replacing several reactors into one group with effective baseline and
power values. You can disable it with ``--no-reactor-groups``

Let's run and explore! Create a juno instance and drop into repl:

.. code-block:: ipython

  python ./gna juno --name juno -- repl
  
  In [1]: ns = self.env.ns('juno')

We named our instance of juno as ``juno``, so our first input to the
repl gets ease access to its namespace. Now let's see what's there:

.. code-block:: ipython

  In [2]: [path for path, obs in ns.walkobservables()]
  Out[2]: ['juno/AD1']

We have used ``ns.walkobservables()`` function that returns iterable
over (full path, observable) pairs. As for result, we have only one
observable, spectrum in the only detector (check it with
``self.env.get('juno/AD1').data()``). This is the only histogram, that
is intended for experimental analysis. But it is likely that you want to
take a look into intermediate results of the computations. The bad
news there is no good generic way to navigate the computation
graph and see every step of computation. The good news is it's not
hard to provide additional interesting outputs in the implementation
experiment along with the main observable. Currently it's done by
``ns.addobservable(..., export=False)`` that just stores an additional
flag, indicating that the observable is *internal* and should not be
visible unless explicitly requested. In order to iterate over them too use 
the following snippet:

.. code-block:: ipython

  In [3]: [path for path, obs in ns.walkobservables(internal=True)]
  Out[3]:
  ['juno/AD1_unoscillated',
   'juno/AD1_noeffects',
   'juno/AD1',
   'juno.detectors.AD1/flux_U235',
   'juno.detectors.AD1/flux_U238',
   'juno.detectors.AD1/flux',
   'juno.detectors.AD1/flux_Pu241',
   'juno.detectors.AD1/Enu',
   'juno.detectors.AD1/oscprob',
   'juno.detectors.AD1/xsec',
   'juno.detectors.AD1/flux_Pu239',
   'juno.detectors.AD1.HZ/oscprob',
   'juno.detectors.AD1.group0/oscprob',
   'juno.detectors.AD1.DYB/oscprob']

You can access all of them in the same way as other observables. For
example, to plot the final observable spectrum with oscillations 
and unoscillated spectrum you can run the following command:

.. code-block:: bash

  python gna juno --name juno \
            -- spectrum --plot juno/AD1 --plot juno/AD1_unoscillated

Not all of that outputs are histograms, most of them are not integrated
functions of neutrino energy:

.. code-block:: ipython

  In [3]: [(path, obs.datatype().kind) for path, obs in ns.walkobservables(internal=True)]
  Out[3]:
  [('juno/AD1_unoscillated', 2),
   ('juno/AD1_noeffects', 2),
   ('juno/AD1', 2),
   ('juno.detectors.AD1/flux_U235', 1),
   ('juno.detectors.AD1/flux_U238', 1),
   ('juno.detectors.AD1/flux', 1),
   ('juno.detectors.AD1/flux_Pu241', 1),
   ('juno.detectors.AD1/Enu', 1),
   ('juno.detectors.AD1/oscprob', 2),
   ('juno.detectors.AD1/xsec', 1),
   ('juno.detectors.AD1/flux_Pu239', 1),
   ('juno.detectors.AD1.HZ/oscprob', 1),
   ('juno.detectors.AD1.group0/oscprob', 1),
   ('juno.detectors.AD1.DYB/oscprob', 1)]

We see the type of the data here along with the name. 2 means histogram, 1
-- just an array of points. You can access them their contents in the
same way using ``ns.get(...).data()``, but you can't plot them with
``spectrum`` because they have no bins information. There is no
command line command to plot them (although you can invent one) mainly
because it's not easy to handle general case and plotting them via
REPL or small ad hoc script is almost the same amount of typing
with way more flexibility without worrying about uninteresting
things. 

Let's plot something, for example, cross section:

.. code-block:: ipython

  In [3]: Enu = self.env.get('juno.detectors.AD1/Enu').data()

  In [4]: xsec = self.env.get('juno.detectors.AD1/xsec').data()

  In [5]: plt.plot(Enu, xsec)
  Out[5]: [<matplotlib.lines.Line2D at 0x7fea400b8f50>]

  In [6]: plt.show()

You should have the picture after the last command. Of course you can
save it with ``plt.savefig(filepath)`` instead of showing, or use any
other matplotlib commands, like setting labels, grids, colors, legends
and so on. You can plot the total flux, isotope fluxes and
oscillations probability for each reactor (reactor group) pair in
absolutely the same way.

As you can see, we had requested values of both x and y axes from the
system, instead of specifying the :math:`E_{\bar{\nu}}` points as inputs to
the cross section function. Unfortunately, there is even no clear
way to get value of xsec in arbitrary points. That's drawback of the
design, the calculating object is tightly bound to the outputs of
another objects, that trying to feed something else to it will cause
changes to the whole computational graph and likely fail. It's
possible to overcome this modifications in core or by reconstructing
the interesting part of the graph in separate script (with the help of
copy-paste approach). On the other hand, you can to look at that
limitation in a positive way: you can get only such a plots, that are
*really* used in the final computation, thus making your papers more
honest.

The other consequence of that design is that you don't always get the
exact function you may be interested in. For example, let's try to run
juno with the first order IBD:

.. code-block:: ipython
   
  python ./gna juno --name juno --ibd first -- repl
  
  In [1]: Enu = self.env.get('juno.detectors.AD1/xsec').data()

  In [2]: Enu.shape
  Out[2]: (400, 5)

Now ``Enu`` (and others, that depends on ``Enu``, like ``xsec`` or
``oscprob``) is two dimensional array (while with default zero order
IBD it was one dimensional, check it). That's because now :math:`E_\nu`
depends not only on :math:`E_e` which (we have as input) but also on
the :math:`\cos\theta` and the integration is two
dimensional. Actually the integration over :math:`\cos\theta` is now
hardcoded to be of the fifth order, hence the 5 in the shape. You can
get that ``ctheta`` values with:

.. code-block:: ipython

  In [3]: self.env.get('juno.detectors.AD1/ctheta').data()
  Out[3]: array([-0.90617985, -0.53846931,  0.        ,  0.53846931,
  0.90617985])

You can plot everything, for example, for :math:`\cos\theta = 0` by
using ``Enu[:, 2]`` and ``xsec[:, 2]`` (because ``ctheta[2] == 0``),
but suppose your integration code is more intelligent and uses
different number of ``ctheta`` points for each energy bin. The array
in that case will likely be unshaped and to access it more complicated
values selection will be required. That's one of the reasons why
there is no generic plotting command implemented since it needs a special care.

If you don't like default binning, you can change it with
``--binning``. Four parameters should be passed -- detector name (here
we have only ``AD1``), minimal, maximal values of the visible energy
and bins count. Only uniform binning is supported in the interface,
you can always extend it and pass arbitrary ``edges`` to the
``Detector`` during initialization.

Of course there is also a number of parameters you can work with. You
can see all the available values with:

.. code-block:: ipython

  In [4]: [name for name, value in self.env.ns('juno').walknames()]
  Out[4]:
  ['juno.isotopes.Pu239.EnergyPerFission',
   'juno.isotopes.U238.EnergyPerFission',
   'juno.isotopes.Pu241.EnergyPerFission',
   'juno.isotopes.U235.EnergyPerFission',
   ...

Not all of them are really parameters that you can change, some are values
that depend on other values and gets recalculated on demand. For
example, there are two values for larger square mass difference:

.. code-block:: ipython

  In [5]: self.env.get('juno.oscillation.DeltaMSqEE')
  Out[5]: <ROOT.GaussianParameter<double> object at 0x55da98da7860>

  In [6]: self.env.get('juno.oscillation.DeltaMSqEE')
  Out[6]: 0.00234

  In [7]: self.env.get('juno.oscillation.DeltaMSq23')
  Out[7]: <ROOT.Variable<double> object at 0x55da99544580>

  In [8]: self.env.get('juno.oscillation.DeltaMSq23').value()
  Out[8]: 0.0022877478

The ``DeltaMSqEE`` is the real independent parameter (hence the type,
``GaussianParameter``), while the ``DeltaMSq23`` is just read only
value (typed as ``Variable``). Their origin is also different:
independent parameters are defined by the ``ns.reqparameter()`` or
``ns.defparameter()`` calls, while the read-only values originate from
evaluables defined in ``GNAObject``\s. For example, there is class
``OscillationExpressions`` which provides expressions to calculate
values on base of others. Whenever an object requires a variable (by
using ``GNAObject::variable_``) and there is no corresponding
independent parameter available, the binding system will try to get
it's value by using available iterables. In case of ``DeltaMSq23``,
which is required by the oscillations probability class
``OscProbPMNS``, the expression to get ``DeltaMSq23`` from
``DeltaMSqEE``, ``Alpha``, ``SinSq12`` and ``DeltaMSq12`` was
automatically used.

You can't change ``Variable`` objects directly, there is even no
``set()`` method. But if you change the independent parameters which
they depend on, the values will be automatically updated:

.. code-block:: ipython

  In [9]: self.env.get('juno.oscillation.DeltaMSqEE').set(2.5e-3)

  In [10]: self.env.get('juno.oscillation.DeltaMSq23').value()
  Out[10]: 0.0024477478

  In [11]: self.env.get('juno.oscillation.Alpha').set('inverted')

  In [12]: self.env.get('juno.oscillation.DeltaMSq23').value()
  Out[12]: 0.0025522522

There is currently no user friendly way to switch the set of
independent parameters. All you need to done is to ensure that they
are created prior to first usage by computational object, and to do
this, you can for example directly change
``gna.parameters.oscillation`` or add rather some more nice command
line arguments switches.
