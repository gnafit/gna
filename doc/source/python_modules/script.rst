Script
^^^^^^

`Script` module is used for the doing some job that doesn't need a dedicated
module such as setting a number of parameters to a specific values, dumping
observables to files, reading dataset + cleanup from files and whatever you
see fit. The script can be arbitrary valid Python code and can access parts of
GNA computational graphs that have been setup before calling script.

The module accepts the following **positional** arguments:
    * ``file`` -- the path to the Python script to be executed.
    * ``args`` -- the sequence of arguments that can be passed to the script
      body with index-based access.

See for examples of valid scripts in ``scripts/ui``. 
The CLI usage:

.. code-block:: bash

   python gna -- script ./path/to/script.py arg0 arg1 arg2

The parameters arg0, arg1, arg2 can be accessed from the script body as
``self.opts.args[0], self.opts.args[1], self.opts.args[2]``
