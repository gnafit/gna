.. _TransformationBundle:

TransformationBundle
^^^^^^^^^^^^^^^^^^^^

``TransformationBundle`` is a base class to implement a concept of bundles. Bundle is needed to simplify the following
tasks:

+ Construction and configuration a single transformation
+ Construction, configuration and connection of several transformations. In this sense Bundle is a transformation of a
  higher level.
+ Based on the given configuration initialize necessary environments and variables, set uncertainties, etc.

``TransformationBundle`` is defined by overriding the following members and methods:
    name (str): bundle name, used to request the bundle configuration and create the storage namespace
    build(method): the method which builds the transformations chain. The class is expecte


