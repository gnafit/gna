Making sense
------------

Let's try log the classes and their meaning and usage.
The documents and relations may then be converted to doxygen.

General notes
^^^^^^^^^^^^^

* The virtual inhertance is implemented via CRTP_ mechanism.

.. _CRTP: https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern

Glossary
^^^^^^^^

Trasnformation, trasnformation node, calculation node, node
    calculation unit with any number of inputs and at least one output

Input, argument, source
    calculation node input data

Output, return value, sink
    calculation node output data

Connection, connect
    precedure of the connection between one Entry_'s output to the other Entry_'s input

Undocumented classes
^^^^^^^^^^^^^^^^^^^^

.. _Initializer:

Initializer : class template
    * used to initialize transformation via CRTP chain
    * created via GNAObject_::Base_::transformation\_
    * creates Entry_ instance and assignes it to the caller

.. _GNAObject:

GNAObject : class

.. _GNASingleObject:

GNASingleObject : class

.. _Transformation:

Transformation : class template
    * blablabla (TBD)
    * contains
      + list of MemFunction_ instances
      + list of MemTypesFunction_ instances
    * CRTP_ base
    * requires ancestor to also inherit GNAObject_

Also see Errors_

TransformationBase header
^^^^^^^^^^^^^^^^^^^^^^^^^

.. _SingleOutput:

SingleOutput : class
    * copmlements Base_ class
    * used for the cases when there is only one output
    * parent to GNASingleObject_

TransformationTypes namespace (TransformationBase)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Functions
"""""""""

.. _MemFunction:

MemFunction : std::function
    * template
    * (T* this, Args_, Rets_)
    * an implementation of the particular transformation

.. _MemTypesFunction:

MemTypesFunction : std::function
    * template
    * (T*, Atypes_, Rtypes_)
    * an transformation input/output types initialization and checking

Main classes
""""""""""""

.. _Base:

Base : class
    * base class for GNAObject_
    * contains list of Entry_ instances
    * accessed via Accessor_ class
    * may be connected
    * implements transformation\_ member function used to define any transformation (returns Initializer_ instance)

.. _Entry:

Entry : struct
    * base class fore the calculation node representation
    * has methods to:

      + add sources/sinks

      + evaluate/update types/values

      + freeze/unfreeze/touch

    * gives access to:

      + sources/sinks

      + data

      + taint flag

    * accessed via Handle_ class
    * named

Indirect access classes
"""""""""""""""""""""""

.. _Accessor:

Accessor : class
    * limited indirect access to Base_ instance
    * access to Entry_ by name or index via Handle_

.. _Args:

Args : struct
    * limited indirect access to Entry_ instance
    * transformation input implementation
    * access to Entry_'s data

.. _Atypes:

Atypes : struct
    * limited indirect access to Entry_ instance
    * used for inputs' type checking
    * access to Entry_'s ``DataType``

.. _Handle:

Handle : class
    * indirect access to Entry_ instance
    * implements and redirects Entry_ methods

.. _InputHandle:

InputHandle : class
    * limited indirect access to Source_
    * may be connected to OutputHandle_

.. _Rets:

Rets : struct
    * limited indirect access to Entry_ instance
    * transformation output implementation
    * access to Entry_'s data
    * may be (un)frozen

.. _Rtypes:

Rtypes : struct
    * limited indirect access to Entry_ instance
    * used for outputs' type checking
    * access to Entry_'s ``DataType``

.. _Sink:

Sink : struct
    * public indirect access to Entry_ instance
    * named

.. _Source:

Source : struct
    * public indirect access to Entry_ instance
    * may be connected to Sink_ instance
    * named

.. _OutputHandle:

OutputHandle : class
    * limited indirect access to Sink_
    * may be:
      + checked if depends on ``changable`` instance

Errors
""""""

.. _TypeError:

TypeError : class

.. _CalculationError:

CalculationError : class

.. _SinkTypeError:

SinkTypeError : class

.. _SourceTypeError:

SourceTypeError : class

