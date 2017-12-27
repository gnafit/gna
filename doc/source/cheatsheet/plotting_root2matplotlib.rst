Plotting ROOT objects with matplotlib
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following code block defines some useful methods to plot ROOT objects.

.. code-block:: python

   from mpl_tools import bind

For the detailed instructions refer to the matplotlib_ documentation and reference_ to relevant functions (plot, bar, etc).

.. _matplotlib: http://matplotlib.org/contents.html
.. _reference: http://matplotlib.org/api/pyplot_api.html?highlight=pyplot#module-matplotlib.pyplot

.. table::  `TH1`
   :widths: 80 100

   +-----------------------------------------+--------------------------------------------------------------------------+
   | ``hist1.plot(...)``                     | plot TH1* using ``pyplot.plot(x, y, ...)`` function (x,y are overridden) |
   +-----------------------------------------+--------------------------------------------------------------------------+
   | ``hist1.plot(autolabel=True, ...)``     | same as above, but guess label from histogram's title                    |
   +-----------------------------------------+--------------------------------------------------------------------------+
   | ``hist1.errorbar(autolabel=True, ...)`` | plot TH1* using ``pyplot.errorbar(x, y, yerr, xerr, ...)``               |
   +-----------------------------------------+--------------------------------------------------------------------------+
   | ``hist1.bar(autolabel=True, ...)``      | plot TH1* using ``pyplot.bar(x, y, yerr, xerr, ...)``                    |
   +-----------------------------------------+--------------------------------------------------------------------------+
   || ``hist1a.bar(divide=3, shift=0, ...)`` | plot three TH1* using ``pyplot.bar()``. The width of the bins will be    |
   || ``hist1b.bar(divide=3, shift=1, ...)`` | times shorter and the bins will be displaced and not overlapped          |
   || ``hist1c.bar(divide=3, shift=2, ...)`` |                                                                          |
   +-----------------------------------------+--------------------------------------------------------------------------+


``autolabel`` may be applied to all above commands.
   

.. table::  `TH2`
   :widths: 80 100

   +-----------------------------------------+---------------------------------------------------------------------------------+
   | ``hist2.pcolorfast(...)``               | plot TH2* ``ax.pcolorfast()`` (fastest, each axis should have equal width bins) |
   +-----------------------------------------+---------------------------------------------------------------------------------+
   | ``hist2.pcolorfast(colorbar=True,...)`` | same as above, adds a colorbar                                                  |
   +-----------------------------------------+---------------------------------------------------------------------------------+
   | ``hist2.pcolorfast(mask=0.0,...)``      | same as the first command. Do not colorize bins with value=0.0 (ROOT behaviour) |
   +-----------------------------------------+---------------------------------------------------------------------------------+
   | ``hist2.pcolormesh(...)``               | plot TH2* ``ax.pcolormesh()`` (slower than pcolorfast)                          |
   +-----------------------------------------+---------------------------------------------------------------------------------+
   | ``hist2.imshow(...)``                   | plot TH2* ``ax.imshow()``                                                       |
   +-----------------------------------------+---------------------------------------------------------------------------------+

``colorbar`` and ``mask`` options may be applied to all above commands.

.. table::  `TGraph`
   :widths: 80 100

   +-------------------------+--------------------------------------------------------------------------------------+
   | ``graph.plot(...)``     | plot ``TGraph``/``TGraphErrors``/``TGraphAsymmErrors`` via ``plt.plot()``, no errors |
   +-------------------------+--------------------------------------------------------------------------------------+
   | ``graph.errorbar(...)`` | plot ``TGraphErrors``/``TGraphAsymmErrors`` via ``plt.errorbar()``                   |
   +-------------------------+--------------------------------------------------------------------------------------+

