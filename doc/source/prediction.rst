Accessing calculated predictions
==================================

Once you have constructed a computational chain, you can get access to
actual computation results with ``data()`` method, and its type with
``datatype()``. Let's try a few examples inside the REPL::

  $ python ./gna gaussianpeak --name peak1 -- repl
  ...
  In [1]: obs = self.env.ns('peak1').observables['spectrum']

  In [2]: obs.data()
  Out[2]: 
  array([ 0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25,
          0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25,
          0.25,  0.25])

The first command gives access to the observable we defined in
``gaussianpeak.py`` as the last step -- we request the namespace
``peak1`` (as specified in ``--name`` of ``gaussianpeak`` command) from
``env`` (``self.env`` is set to the same global ``gna.ui.env``) and
access the observable named ``spectrum`` from there. In the second
line we get the data contained there -- just the contents of the
histogram bins. They are all the same, as we have defined our default
value of :math:`\mu` to be zero, so no peak, just flat
background. Let's try to set :math:`\mu` to something more
interesting::

  In [3]: self.env.ns("peak1")['Mu'].set(1)

  In [4]: obs.data()
  Out[4]:
  array([ 0.25003941,  0.25273751,  0.29447097,  0.42635878,  0.42635878,
          0.29447097,  0.25273751,  0.25003941,  0.25000013,  0.25      ,
          0.25      ,  0.25      ,  0.25      ,  0.25      ,  0.25      ,
          0.25      ,  0.25      ,  0.25      ,  0.25      ,  0.25      ])

Once we changed the parameter ``Mu`` inside the namespace ``peak1``,
the data is automatically updated after next call to ``data()`` and we
see the changes.

If you want to plot our histogram, you'd like to know the
corresponding binning. The binning is a part of the Histogram ``datatype``
and it's defined by the edges. You can access them with the following
syntax::

  In [5]: import numpy as np

  In [6]: edges = np.array(obs.datatype().hist().edges())

  In [7]: edges
  Out[7]:
  array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ,  1.25,  1.5 ,  1.75,  2.  ,
          2.25,  2.5 ,  2.75,  3.  ,  3.25,  3.5 ,  3.75,  4.  ,  4.25,
          4.5 ,  4.75,  5.  ])

Please note that the ``edges()`` method is implemented in ``C++`` and returns
``std::vector<double>``. Since (currently) no pythonization is done
for it, you'll need to call ``np.array()`` explicitly to get a nice
representation of the values. Having both the bin edges and contents we
can plot it with ``matplotlib``::

  In [8]: from matplotlib import pyplot as plt

  In [9]: plt.step(edges[:-1], obs.data())
  Out[9]: [<matplotlib.lines.Line2D at 0x7f7aabf684d0>]

  In [10]: plt.show()

Alternatively you can get the similar plot from the command line::

  python ./gna gaussianpeak --name peak1 -- ns --value peak1.Mu 1 -- \
  spectrum --plot peak1/spectrum

Here we have set up the same theoretical model ``peak``, set up
the parameter ``Mu`` value to 1 with the ``ns`` command and finally
used the ``spectrum`` command (which basically just plots a histogram
of the given observable) to produce the plot. The ``peak1.Mu`` is the
full path to parameter ``Mu`` inside namespace ``peak1``, and
``peak1/spectrum`` is the path to observable ``spectrum`` inside the
same namespace. Please check the code in ``pylib/gna/ui/spectrum.py``
to see how it's plotted and extend it with more features.
