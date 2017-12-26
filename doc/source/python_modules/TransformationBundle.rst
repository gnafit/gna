.. _TransformationBundle:

TransformationBundle
^^^^^^^^^^^^^^^^^^^^

Overview
""""""""

``TransformationBundle`` is a base class to implement a concept of bundles. Bundle is needed to simplify the following
tasks:

+ Construction and configuration a single transformation.
+ Construction, configuration and connection of several transformations. In this sense Bundle is a transformation of a
  higher level.
+ Based on the given configuration initialize necessary environments and variables, set uncertainties, etc.

Design principle
""""""""""""""""

Bundles define the way configuration files are processed and variables are created, therefore bundles are meant to be
persistent during GNA development cycle.

Bundle name usually contains a version mark. It is guaranteed that the bundle with given name and version will not be
modified in a way that breaks the backwards compatibility.

If significant modification is required another bundle is created with different version.

Initialization
""""""""""""""

There are two ways to initialize a bundle:

1. By using method ``execute_bundle`` (preferred). One can find example of creating bundle instance of
   :ref:`detector_iav_db_root_v01`:

   .. code-block:: python

       from gna.bundle import execute_bundle
       bundle = execute_bundle( name='detector_iav_db_root_v01', cfg=cfg, namespaces=namespaces, common_namespace=common_namespace, storage=storage )
       bundle = execute_bundle( cfg=cfg, namespaces=namespaces, common_namespace=common_namespace, storage=storage )

   if name is not passed, it's read from ``cfg.bundle``. The specified class ``<name>`` is searched in the module
   ``gna.bundles.<name>``.

2. By direct class instantiation:

   .. code-block:: python

       from gna.bundles.detector_iav_db_root_v01 import detector_iav_db_root_v01
       bundle = detector_iav_db_root_v01( cfg=cfg, namespaces=namespaces, common_namespace=common_namespace, storage=storage )

Arguments
"""""""""

The arguments to the bundle initialization are the following:

1. ``cfg`` — the configuration of type  :ref:`NestedDict`. The configuration items may be found in the documentation of
   specific bundle.

2. ``common_namespace`` — :ref:`namespace <environment_ns>`, where the common variables will be initialized. By default
   is set to ``env.globalns``.

3. ``namespaces`` — a list of namespaces or namespace names (from ``common_namespace``), where uncorrelated variables
   will be stored. The bundle is expected to create an output transformation for each namespace. By default set to
   ``[common_namespace]``.

4. ``storage`` — namespace :ref:`namespace <environment_ns>`, where bundle will store created transformations and stuff.
   The bundle will create ``storage[self.name]`` sub namespace to keep its stuff.

Execution and output
""""""""""""""""""""

The transformation is defined by two methods ``define_variables`` and ``build``, defined in individual bundles. First
method reads the configuration and uncertain parameters within ``common_namespace`` and ``namespaces``. Second method
creates a single transformation or a graph of the transformations. The graph has an output transformation for each
namespace in ``namespaces``. After the execution the following tuples are populated:

1. ``self.output_transformations`` — computation graph output transformations (for each ``namespace``).
2. ``self.outputs`` — transformations outputs (for each ``namespace``).
3. ``self.inputs`` — inputs, that should be connected in order graph was completely initialized.

