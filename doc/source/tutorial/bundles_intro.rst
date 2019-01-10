Bundles configuration and seach path
''''''''''''''''''''''''''''''''''''

Bundle classes are loaded automatically. The default bundle search path is ``pylib/gna/bundle/``. The bundle search path
is a configured option, which is defined in the configuration file ``config/gna/gnacfg.py``:

.. code-block:: python

    bundlepaths = [ './pylib/gna/bundles', './pylib/gna/bundles_legacy' ]

The configuration file is included into git repository and should not be modified. Still there is a possibility to
override it by providing a local configuration file ``config_local/gna/gnacfg.py``:

.. code-block:: sh

    mkdir -p config_local/gna
    echo "bundlepaths = [ './pylib/gna/bundles', './pylib/gna/bundles_legacy', '../macro/tutorial/bundles' ]" > config_local/gna/gnacfg.py

The configuration from the `local` version is loaded after the global one and overrides its settings. After creating the
local configuration file run `gna` without arguments:

.. code-block:: sh

   ./gna

The output should indicate that local configuration is loaded:

.. code-block:: text

   Loading config file: config/gna/gnacfg.py
   Loading config file: config_local/gna/gnacfg.py

As one may notice, we have added an extra path to search for bundles. The path contains tutorial bundles.

A bundle is specified by two strings: `name` and `version`. The full name is then produced by joining name and version
with `_` used as a separator. For example for bundle `parameters` with version `ex01` a python file `parameters_ex01`
will be searched in each of the bundle search paths. The file should define a class with the same name
(`parameters_ex01`) which will be loaded and instantiated.

A typical bundle configuration is a dictionary of the following form:

.. code-block:: python

    cfg = NestedDict(
      bundle = 'parameters_ex01',
      option1 = value1,
      option2 = value2,
      ...
    )

The bundle is executed calling a method ``execute_bundle(cfg)``, which finds relevant bundle class, instantiates it with
configuration provided and executes it. In a sense bundle is a function with a lot of arguments.

Are more complex version of the configuration looks like this:

.. code-block:: python

    cfg = NestedDict(
      bundle = dict(
                     name='parameters',
                     version='ex01',
                     ...
                     ),
      option1 = value1,
      option2 = value2,
      ...
    )

A `bundle` field is a dictionary which specifies which bundle should be used to parse the configuration. This dictionary
may contains some other options. The distinction between configuration options and bundle options are the following:
bundle options are parsed by the GNA similarly for all the bundles and may be sometimes overridden. The configuration
options are unique to the particular bundle of the particular version and may be parsed only by it.

