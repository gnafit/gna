Miscellaneous
~~~~~~~~~~~~~

Changing random seed
^^^^^^^^^^^^^^^^^^^^

To change the seed of the random generators used in ``NormalToyMC``,
``CovarianceToyMC`` and ``PoissonToyMC`` use the following code:

from C++:

.. code:: C++

   #include "Random.hh"
   GNA.Random.seed( seed )

or from python:

.. code:: python

    import ROOT as R
    R.GNA.Random.seed( seed )

Note, if you are using ``numpy`` random generator, the seed is changed by:

.. code:: python

    import numpy as N
    N.random.seed( seed )

.. caution::

    When using both ``numpy`` and ``GNA`` random generators, make sure
    that they use *different* random seeds. Since they both use Mersenne Twister
    random generator, but different instances, setting similar seed will lead
    random distributions to be based on the same pseudo-random sequence.

