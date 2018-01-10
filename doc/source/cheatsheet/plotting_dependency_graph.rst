Plotting dependency graph
^^^^^^^^^^^^^^^^^^^^^^^^^

The dependency graph may be visualized by `graphviz <http://www.graphviz.org>`_. The module ``gna.graphviz`` is used for
it. It requires ``pygraphviz`` python module to be installed in the system.

There are two ways to get the dependency graph: via gna UI module ``gna.ui.graphviz`` or explicitly in the python
script.

Using graphviz in a python script
"""""""""""""""""""""""""""""""""

The example from the ``tests/bundle/detector_dbchain.py`` (line 110) may be found below:

.. code-block:: python

    try:
        from gna.graphviz import GNADot
        graph = GNADot( b.output_transformations[0][0] )
        graph.write('output/graph.dot')

The module provides the only class ``GNADot`` that accepts the transformation as the only argument. The method ``write``
is used to save the graph to ``output/graph.dot``.

Using graphviz as UI module
"""""""""""""""""""""""""""

The ``graphviz`` module may be used to plot the graph for existing observable as in example:

.. code-block:: bash

    ./gna \
        -- juno --name juno_nh \
        -- graphviz juno_nh/AD1 -o output/juno.dot

Plotting the .dot file
""""""""""""""""""""""

The output file may be plotted via ``xdot``. For example:

.. code-block:: bash

    xdot output/juno.dot










