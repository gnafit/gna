.. _Constructors:

Constructors
^^^^^^^^^^^^

The python module ``constructors`` implements some simplified constructors for C++ objects. See also ``conversion``
module.

.. table::  Constructors
   :widths: 100 80

   +----------------------------------+------------------------------------------------------------+
   | ``stdvector(array)``             | ``std::vector`` object from an array                       |
   +----------------------------------+------------------------------------------------------------+
   | ``Points(array)``                | object from an array                                       |
   +----------------------------------+------------------------------------------------------------+
   | ``Histogram(edges, data)``       | object from arrays                                         |
   +----------------------------------+------------------------------------------------------------+
   | ``Rebin(edges, rounding)``       | object from edges array and rounding                       |
   +----------------------------------+------------------------------------------------------------+
   | ``GaussLegendre(edges, orders)`` | integrator with edges array and order(s) (array or number) |
   +----------------------------------+------------------------------------------------------------+
