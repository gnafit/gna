pargrid
"""""""

The module provides tools for creating grids to scan over parameters.

It supports `range`, `linspace`, `logspace` and `geomspace` which are similar to their analogues from `numpy`.
It also supports a list of values passed from the command line.

**Positional arguments**

    * ``name`` -- define name if the parameter grid

**Options**

    * ``--range`` -- define grid via NumPy arange (*endpoint not included*)

        + Positional arguments: *PARAMETER* *START* *STOP* *STEP*


    * ``--logspace`` -- define grid via NumPy logspace (*endpoint included*)

        + Positional arguments: *PARAMETER* *START_POWER* *STOP_POWER* *NUMBER*


    * ``--geomspace`` -- define grid via NumPy geomspace (*endpoint included*)

        + Positional arguments: *PARAMETER* *START* *STOP* *NUMBER*


    * ``--linspace`` -- define grid via NumPy linspace (*endpoint included*)

        + Positional arguments: *PARAMETER* *START* *STOP* *NUMBER*


    * ``--list`` -- define grid via NumPy array of values

        + Positional arguments: *PARAMETER* *SPACE SEPARATED VALUES* (*VALUE_1* *VALUE_2* *VALUE_3*...)


    * ``--segments`` -- define a segmentation of grid to allow parallel scanning of large grids

        + Positional arguments: *NUMBER OF SEGMENTS* *CURRENT SEGMENT*


    * ``-v, --verbose`` -- define verbosity level


**Examples**

    * Generate a linear grid for the parameter 'E0':

        .. code-block:: bash

            ./gna \
                -- gaussianpeak --name peak \
                -- pargrid scangrid --linspace peak.E0 0.5 4.5 10 -vv


    * Provide a list of grid values from a command line:

        .. code-block:: bash

            ./gna \
                -- gaussianpeak --name peak \
                -- pargrid scangrid --linspace peak.E0 1 2 8 -vv


See also: *minimizer-scan*.