Plotting outputs via matplotlib 2d: plots and histograms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GNA defines a set of convenience methods to plot the transformation outputs with matplotlib. The methods are wrappers
for regular `matplotlib <https://matplotlib.org/api/pyplot_api.html>`_ commands. For the complete matplotlib
documentation please refer the official `site <https://matplotlib.org/api/pyplot_api.html>`_.

Plotting arrays
"""""""""""""""

A ``plot(...)`` method is defined implementing
`plot(y, ...) <https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot>`_ call passing output contents as `y`.
The method works the same way for both arrays and histograms.

.. literalinclude:: ../../../macro/tutorial/basic/04_points_plot.py
    :linenos:
    :lines: 4-31,38
    :emphasize-lines: 25,26
    :caption: :download:`04_points_plot.py <../../../macro/tutorial/basic/04_points_plot.py>`

When 1d array is passed (line 25) it is plotted as is while for 2d array (line 6) each column is plotted in separate.
The latter produces the blue line on the following figure while the former produces orange, green and red lines.

.. figure:: ../../img/tutorial/04_points_plot.png

    A example ``output.plot()`` method for outputs with 1d and 2d arrays.

.. table:: Keyword options

    +------------------+---------------------------------+
    | `transpose=True` | transpose array before plotting |
    +------------------+---------------------------------+

Plotting arrays vs other arrays
"""""""""""""""""""""""""""""""

If `X` vs `Y` is desired ``output_y.plot_vs(output_x, ...)`` syntax may be used. Matplotlib
`plot(x, y, ...) <https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot>`_ function is used.

The twin method ``output_x.vs_plot(output_y, ...)`` may be used in case reversed order is desired.

.. literalinclude:: ../../../macro/tutorial/basic/05_points_plot_vs.py
    :linenos:
    :lines: 4-27,34
    :emphasize-lines: 22
    :caption: :download:`05_points_plot_vs.py <../../../macro/tutorial/basic/05_points_plot_vs.py>`

.. figure:: ../../img/tutorial/05_points_plot_vs.png

    A example ``output_x.plot_vs(output_y)`` method for outputs.

.. table:: Keyword options

    +------------------+----------------------------------+
    | `transpose=True` | transpose arrays before plotting |
    +------------------+----------------------------------+

Plotting histograms
"""""""""""""""""""
..
    The
    A ``matshow(...)`` method is defined implementing
    `matshow(A, ...) <https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.matshow>`_ call passing output contents as `A`.
    This method works with both histograms and arrays. When applied to histograms it ignores bin edges definitions and plots
    the matrix anyway.

.. literalinclude:: ../../../macro/tutorial/basic/07_hist_plot.py
    :linenos:
    :lines: 4-43,49
    :emphasize-lines: 25,37
    :caption: :download:`07_hist_plot.py <../../../macro/tutorial/basic/07_hist_plot.py>`

.. figure:: ../../img/tutorial/07_hist_plot.png

    A example ``output.plot_hist()`` and ``output.bar()`` method for outputs.

Overlapping histograms
""""""""""""""""""""""

.. literalinclude:: ../../../macro/tutorial/basic/08_hists_plot.py
    :linenos:
    :lines: 4-22,27-40,75
    :emphasize-lines: 29-31
    :caption: :download:`08_hists_plot.py <../../../macro/tutorial/basic/08_hists_plot.py>`

.. figure:: ../../img/tutorial/08_hists_plot_hist.png

.. literalinclude:: ../../../macro/tutorial/basic/08_hists_plot.py
    :linenos:
    :lines: 43-56,75
    :emphasize-lines: 10-12
    :caption: :download:`08_hists_plot.py <../../../macro/tutorial/basic/08_hists_plot.py>`

.. figure:: ../../img/tutorial/08_hists_plot_bar1.png

.. literalinclude:: ../../../macro/tutorial/basic/08_hists_plot.py
    :linenos:
    :lines: 59-72,75
    :emphasize-lines: 10-12
    :caption: :download:`08_hists_plot.py <../../../macro/tutorial/basic/08_hists_plot.py>`

.. figure:: ../../img/tutorial/08_hists_plot_bar2.png
