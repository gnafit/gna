Oscillation Probabilities
~~~~~~~~~~~~~~~~~~~~~~~~~

Class OscProbPMNS
^^^^^^^^^^^^^^^^^
Description
===========

Calculates the exact 3-:math:`\nu` oscillation probabilities in vacuum using the expressions
for the elements of the PMNS matrix. There a few options how it can be done:
    1. By splitting oscillation probability into energy dependant components
       (see JUNO docs) and summing the components with weights after
       integration is done. The following trasnformation are used:
        * ``comp12``, component related to :math:`\Delta m^2_{12}`;
        * ``comp13``, component related to :math:`\Delta m^2_{13}`;
        * ``comp23``, component related to :math:`\Delta m^2_{23}`;
        * ``compCP``, CP-violation related component;
        * ``probsum``, sum of components weighted with elements of PMNS matrix.
    2. ``full_osc_prob`` -- plain full oscillation formula in single
       transformation.
    3. ``average_oscillations`` -- oscillations averaged over energy.
   
Inputs
======
    1. ``comp...`` expects array of energy as inputs
    2. ``probsum`` expects all components + ``comp0`` which is typically
       normed flux of antineutrinos
    3. ``full_osc_prob`` expects array of energy
    4. ``average_oscillations`` expects an flux to be averaged


Outputs
=======
   1. ``comp...`` -- values of components
   2. ``probsum`` -- weighted sum of all components. In case of JUNO or Daya
      Bay those components would be a histograms with number of events.
   3. ``full_osc_prob`` -- array of full oscillation probabilities.
   4. ``average_oscillations`` -- averaged flux.


Class OscProbPMNSMult
^^^^^^^^^^^^^^^^^^^^^

Description
===========

Approximate 3-:math:`\nu` formula that allows one to group close reactors into
one effective reactor to speed up calculations. Oscillation probability is
also splitted into the components. See the  Marrone_.

Inputs
======
The same as for the ``OscProbPMNS`` components and probsum.

Outputs
=======
The same as for the ``OscProbPMNS`` components and probsum.

.. _Marrone: https://arxiv.org/pdf/1309.1638.pdf

Class OscProbPMNSDecoh
^^^^^^^^^^^^^^^^^^^^^^

Description
===========

Exact 3-:math:`\nu` formula in the framework of wavepacket approach. Oscillation probability is
also splitted into the components. See the Akhmedov_ and Daya Bay paper_.

.. _Akhmedov: https://arxiv.org/pdf/0905.1903.pdf
.. _paper: https://arxiv.org/pdf/1608.01661.pdf

Inputs
======
The same as for the ``OscProbPMNS`` components and probsum.

Outputs
=======
The same as for the ``OscProbPMNS`` components and probsum.


Implementation
^^^^^^^^^^^^^^

Each class has to be initialized with initial and final flavor by passing
corresponding ``ROOT.Neutrino`` objects.

    Example:
        ``ROOT.OscProbPMNS(ROOT.Neutrino.amu(), ROOT.Neutrino.ae())``
        initialize oscillation probability from :math:`\bar{\nu_{\mu}}` to
        :math:`\bar{\nu_{e}}`

Each class is derived from a ``OscProbPMNSBase`` that based on initial and
final flavor initialize expressions for required elements of PMNS variables
(see ``neutrino/PMNSVariables.hh``) and mass splittings (see
``neutrino/OscillationVariables.hh``) including effective and exposing it into
the Python environment.

