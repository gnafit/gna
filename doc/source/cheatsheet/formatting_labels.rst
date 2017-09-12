Formatting labels
^^^^^^^^^^^^^^^^^

GNA contains a tool for managing labels which allows:
    * formatting labels based on common phrases
    * formatting variable/unit pairs with possible common divisor

The module loads all the files from ``config/dictionaries`` folder. Each dictionary file
contains dictionary with label=text pairs.

The configuration may be found in ``config/gna/labels.py``:
    * Provides format strings for name/offset/unit specification.
    * Provides common divisors for variables.

The module is loaded via:

.. code:: python

   from gna.labelfmt import formatter as L
   from gna.labelfmt import reg_dictionary

``L`` is needed for formatting purposes, while ``reg_dictionary`` in case
the dictionary is expected to be provided within current script.

The following code initiates inline dictionary with two labels:

.. code:: python

   mydict = dict( mylabel = 'my text', anotherlabel = 'another text')
   reg_dictionary( 'mydict', mydict )

Below one may find simple examples of using formatter:

.. code:: python

   L('  my label: {mylabel}')
   # '  my label: my text'

   L('  two labels: {mylabel} and {anotherlabel}')
   # '  two labels: my text and another text'

   L('  unknown key representation: {unknown}')
   # '  unknown key representation: ?unknown?'

   L('  {1} {0} arguments: {mylabel}', 'positional', 'using')
   # '  using positional arguments: my text'

   L.s('mylabel')
   # 'my text'

   L.base('mylabel')
   # 'my text'

There is a minor sintax to be used for capitalization:

.. code:: python

  L('  capitalize: {^mylabel}')
   # '  capitalize: My label'

Labels to be read may be provided indirectly, via key=value pair or stored in dictionary itself:

.. code:: python

   L('  indirect access to label from key "var": {$var}', var='mylabel')
   # '  indirect access to label from key "var": my text'

   L('  capitalize indirect: {^$var}', var='mylabel')
   # '  capitalize indirect: My label'

The module has capability of managing variable-unit pairs. In this example we are using units defined
in ``config/dictionaries/osc.py``. Notice that the unit should be provided explicitly:

.. code:: python

  dictionary = dict(
      theta13 = r'$\sin^2 2\theta_13$',
      theta13_unit = '',
      dm32 = r'$\Delta m^2_{32}$',
      dm32_unit = r'$\text{eV}^2$',
  )

Variables are supposed to be used for axis formatting and sometimes do have common divisors, which may
be configured via ``config/gna/labels.py`` or provided explicitly:

.. code:: python

   L.u('dm32')
   # '$\\Delta m^2_{32}$ $\\times10^{-3}$, $\\text{eV}^2$'

   L.u('dm32', offset=-5)
   # '$\\Delta m^2_{32}$ $\\times10^{-5}$, $\\text{eV}^2$'

   L.u('dm32', offset=0)
   # '$\\Delta m^2_{32}$, $\\text{eV}^2$'

   L.u('theta13', offset=-2)
   # '$\\sin^2 2\\theta_13$ $\\times10^{-2}$'

