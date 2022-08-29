comment
"""""""

This module may be used to insert comments into the command line.

**Positional arguments**

    * ``comment`` -- define the comment that is printed to the command line


**Examples**

The command will print the arguments upon execution and does nothing more.

.. code-block:: bash

    ./gna \
        -- comment Initialize a gaussian peak with 50 bins \
        -- gaussianpeak --name peak_MC --nbins 50
