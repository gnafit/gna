Utilites
^^^^^^^^

GNA tools
"""""""""

GNA contains a set of helper tools to simplify calling C++ functions from python. The :ref:`Constructors` module contain
wrappers for often used C++ classes' enabling the user to pass numpy arrays and python lists. The following code
converts the python list of strings to ``std::vector<std::string>``:

.. literalinclude:: ../../../macro/tutorial/basic/00_stdvector.py
    :linenos:
    :lines: 4-
    :emphasize-lines: 3
    :caption: :download:`00_stdvector.py <../../../macro/tutorial/basic/00_stdvector.py>`

The code produces the following output:

.. code-block:: text
    :linenos:

    <ROOT.vector<string> object at 0x55b4546e8be0>, ['str1', 'str2', 'str3']

GNA introduces convenience pythonic methods for its types that may be loaded as follows:

.. code-block:: python

    from gna.bindings import common

They include the methods for printing and plotting with matplotlib.

Tutorial options
""""""""""""""""

The tutorial may be executed in batch mode that disables GUI windows. In order to enable it use ``--batch`` command line
option:

.. code-block:: bash

    ./macro/tutorial/basic/04_points_plot.py --batch

The batch mode will be triggered automatically in case ``$DISPLAY`` environment variable is not set.
