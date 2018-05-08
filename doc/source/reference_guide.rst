Reference guide
===============

Overview
--------

.. toctree::
   :glob:
   :maxdepth: 1

   making_sense

Cheat sheet
-----------

.. toctree::
   :glob:
   :maxdepth: 1

   making_sense
   cheatsheet/*

Python modules
---------------

.. toctree::
   :maxdepth: 1
   :glob:

   python_modules/*

Helper classes
--------------

.. toctree::
   :maxdepth: 1
   :glob:

   cppclasses/*

Transformation bundles
----------------------

.. toctree::
   :maxdepth: 1
   :glob:

   python_modules/TransformationBundle.rst
   python_modules/NestedDict.rst
   bundles/*

Transformations
---------------

Evaluables
^^^^^^^^^^

The following transformations do not provide the inputs and outputs. They rather define the new parameters via evaluable
mechanism.

.. toctree::
   :maxdepth: 1
   :glob:

   transformations/VarDiff.rst
   transformations/VarProduct.rst
   transformations/VarSum.rst

Basic types and actions
^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1
   :glob:

   transformations/HistEdges.rst
   transformations/Histogram.rst
   transformations/Points.rst
   transformations/Rebin.rst

Linear algebra
^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1
   :glob:

   transformations/Cholesky.rst
   transformations/FillLike.rst
   transformations/Identity.rst
   transformations/Normalize.rst
   transformations/Product.rst
   transformations/RenormalizeDiag.rst
   transformations/Sum.rst
   transformations/WeightedSum.rst

Stats and calculus
^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1
   :glob:

   transformations/Chi2.rst
   transformations/CovarianceToyMC.rst
   transformations/CovariatedPrediction.rst
   transformations/Covmat.rst
   transformations/Derivative.rst
   transformations/NormalToyMC.rst
   transformations/Poisson.rst

Math functions
^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1
   :glob:

   transformations/SelfPower.rst

Neutrino oscillations
^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1
   :glob:

   transformations/EvisToEe.rst
   transformations/IbdFirstOrder.rst
   transformations/IbdZeroOrder.rst
   transformations/OscillationProbabilities.rst

Detector related
^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1
   :glob:

   transformations/EnergyResolution.rst
   transformations/HistNonlinearity.rst
   transformations/HistSmear.rst

Complete list
^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1
   :glob:

   transformations/*

.. Template

   Identity transformation
   ~~~~~~~~~~~~~~~~~~~~~~~

   Description
   ^^^^^^^^^^^
   Short description

   Inputs
   ^^^^^^

   1) Input 1

   Variables
   ^^^^^^^^^

   Describe variables, that should be located in a current namespace, but not passed as inputs.

   Outputs
   ^^^^^^^

   1) output 1

   Implementation
   ^^^^^^^^^^^^^^

   Implementation details, formulas, etc.


