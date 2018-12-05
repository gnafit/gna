Integration and interpolation: computing things only once
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let us now look at several quite specific transformations that implement integration and interpolation.

Usual work flow to the integration implies that integrand function :math:`f` is passed to the integrating function
which invokes :math:`f` at a set of points implicitly, i.e. the points are hidden from the user. Within GNA our aim is
to keep the explicit data flow, including the case of the integration. Therefore a set of points required for the
integration is computed once and passed as an input array to the integrand transformation or a bundle of transformations.
Such an approach enables the effective usage of GNA lazy and caching nature. As a consequence the precision of the
integration should be chosen by the user beforehand.

The fact, that all the values of the argument, that are required for the computation are passed at once is taken into
account in the transformations' definitions. Thus, for example, the interpolation is split into two transformations.
First transformation determines which segment or spline of the interpolator should be used to interpolate the value of
function for each :math:`x` of the input array. This potentially ineffective procedure is typically done only once. The
actual interpolation is done by the second transformation that knows exactly which segment should be used for each point
of the input.

Let us look in more details.

Integration
"""""""""""

.. toctree::

    integration_intro
    integration_1d

Interpolation (TBD)
"""""""""""""""""""

.. toctree::

   interpolation
