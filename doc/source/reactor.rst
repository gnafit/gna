Reactor experiments
======================

The Gaussian peak example we have used before is a very simple
experiment model. To include some real world experiments to analysis
more work is to be done to implement the whole computational graph
leading to the experimentally observable values. The graph
construction may be designed to be done partially, with separate
modules defining or extending its parts. In the end you'll again have
one (or more) observables, which will be later employed to comparison
with experimental data, and a number of parameters to be fitted or
fixed.

At the moment, the only implemented experimental model which have
physical sense is the reactor experement model. It's quite simple and
straightforward though and does only the primitive event number
computation.

The computation of observable spectrum is generally done with the
following formula:

.. math::
   N_i = \int\limits_{E_i}^{E_{i+1}} d E
   \int dE_\nu \frac{dE}{dE_\nu}\sigma(E_\nu) P(E_\nu) \sum \limits_k
   n_k S_k(E_\nu)

:math:`N_i` is the event number in the :math:`i`-th bin containting
events with energies :math:`[E_i, E_{i+1}]`; :math:`E` and
:math:`E_\nu` is the positron and neutrino energy;
:math:`\sigma(E_\nu)` is the IBD cross section; :math:`P(E_\nu)` is
the oscillation probability; :math:`S_k(E_\nu)` is the neutrino
spectrum of the :math:`k`-th isotope with corresponding normalization
:math:`n_k`.

The vacuum oscillation probability has the following structure:

.. math::
   P(E_\nu) = \sum_j w_j(\theta) P_j(\Delta m^2, E_\nu)

where the mixing angle only dependant weights :math:`w_j` and mass
difference dependant oscillatory terms :math:`P_j` are
factorized. Since only the oscillatory terms are energy dependant,
when doing fits it's more computationally efficient to take the
weights out of the integrals, making the recomputations a lot faster
if only mixing angles are changed. The corresponding formula, which is
implemented in code is:

.. math::
   N_i = \sum_j w_j \int\limits_{E_i}^{E_{i+1}} d E
   \int dE_\nu \frac{dE}{dE_\nu}\sigma(E_\nu) P_j(E_\nu) \sum \limits_k
   n_k S_k(E_\nu)

The :math:`P_j(E_\nu)` functions and other derived :math:`j`-indexed
functions are generally called *components*.

The given formulae are only for the case of one reactor. If there are
several of them, an additional summation inside the integral should be
performed.

All of that is implemented in the `gna.exp.reactor` class. As the most
basic usage you shuld just specify the properties of each detector
and reactor of the experiment. The class will create the required
parameters in given namespace and setup observables for each
reactor. Let's try it with the following example:



