Making sense
------------

Let's try log the classes and their meaning and usage.
The documents and relations may then be converted to doxygen.

General notes
^^^^^^^^^^^^^

* The virtual inhertance is implemented via CRTP_ mechanism.
* See also CppCoreGuidlines_.

.. _CRTP: https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
.. _CppCoreGuidlines: http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines

Glossary
^^^^^^^^

Trasnformation, trasnformation node, calculation node, node
    calculation unit with any number of inputs and at least one output

Input, argument, source
    calculation node input data

Output, return value, sink
    calculation node output data

Connection, connect
    precedure of the connection between one `TransformationBase::Entry`_'s output to the other `TransformationBase::Entry`_'s input

Undocumented classes
^^^^^^^^^^^^^^^^^^^^

.. _Initializer:

.. _GNAObject:

GNAObject : class
    * Inherits `TransformationBase::Base`_

      + carries list of `TransformationBase::Entry`_ instances

      + has transformation_ member function used to initialize transformations

    * Inherits `ParametrizedTypes::Base`_

      + TBD

Also see Errors_

GNAObject header
^^^^^^^^^^^^^^^^

.. _GNASingleObject:

GNASingleObject : class
    * Implements some shorcuts for GNAObject_ with only one output
    * Inerits GNAObject_ and SingleOutput_

.. _Parameters:

Parameters header
^^^^^^^^^^^^^^^^^

The header contains classes representing individual parameters. The base class is changeable_.

Data
""""

.. _inconstant_data:

inconstant_data : struct template
    * elementary data item (usually double) with taint flag propagation
    * may handle a function pointer to calculate the data value
    * may handle a callback function to be called on data change
    * TBD

.. _inconstant_header:

inconstant_header : struct
    * carries information to track dependencies:
        + taintflag_
        + list of taint signal emitters
        + list of taint signal observers
    * base for inconstant_data_
    * TBD

.. _references:

references : struct
    * a container for changeable_ instances
    * used in inconstant_header_
    * why manually allocated? TBD

Variables
"""""""""

Questions:

    * who unallocates memory, allocated via ``new inconstant_header`` or ``new inconstant_data``?


.. _callback:

callback : class
    * TBD

.. _changeable:

changeable : class
    * keeps data as inconstant_data_ (usually inconstant_data_<double>)
    * propagates taintflag_ among denepdants
    * base for variable_

.. _dependant:

dependant : class template
    * inherits evaluable_
    * implements evaluable_, which value depends on other parameters
    * usuall created via ``mkdep()`` function

.. _evaluable:

evaluable : class tempalte
    * inherits variable_
    * a parameter, which value is evaluated with a function

.. _parameter:

parameter : class template
    * inherits variable_
    * may set the variable_'s value

references : class

.. _taintflag:

taintflag : class
    * indicates (cast too bool) if the value/transformation should be recalculated

.. _variable:

variable : class template
    * inherits changeable_
    * base for evaluable_ and parameter_ clasees
    * can return the variable_'s value
    * can not set the varialbe_s value, see parameter_
    * calls updating function if the value is tainted

.. _Parametrized:

Parametrized header
^^^^^^^^^^^^^^^^^^^

.. _EvaluableDescriptor:

EvaluableDescriptor : class

.. _VariableDescriptor:

VariableDescriptor : class

.. ParametrizedTypes:

ParametrizedTypes namespace
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Main classes
""""""""""""

.. _`ParametrizedTypes::Base`:

Base : class
    * base class for GNAObject_
    * contains list of `ParametrizedTypes::Entry`_ instances
    * contains list of `ParametrizedTypes::EvaluableEntry`_ instances
    * contains list of callback_ instances
    * implements variable\_ member function used to define variables
    * TBD

.. _`ParametrizedTypes::Entry`:

Entry : class
    * TBD

.. _`ParametrizedTypes::EvaluableEntry`:

EvaluableEntry : class

Indirect access classes
"""""""""""""""""""""""

EvaluableHandle : class template
    * indirect access to `ParametrizedTypes::EvaluableEntry`_ instance
    * base for EvaluableDescriptor_

VariableHandle : class template
    * indirect access to `ParametrizedTypes::Entry`_ instance
    * base for VariableDescriptor_

.. _TransformationBase:

TransformationBase header
^^^^^^^^^^^^^^^^^^^^^^^^^

.. _SingleOutput:

SingleOutput : class
    * copmlements `TransformationBase::Base`_ class
    * used for the cases when there is only one output
    * parent to GNASingleObject_

.. _Transformation:

Transformation : class template
    * manages MemFunction_ instances
    * contains

      + list of MemFunction_ instances

      + list of MemTypesFunction_ instances

    * lists of functions are filled within Initializer_
    * CRTP_ base for GNAObject_
    * requires ancestor to also inherit GNAObject_

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

.. _`TransformationBase::Base`:

Base : class
    * base class for GNAObject_
    * contains list of `TransformationBase::Entry`_ instances
    * accessed via Accessor_ class
    * may be connected
    * implements transformation\_ member function used to define any transformation (returns Initializer_ instance)

.. _`TransformationBase::Entry`:

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

Initializer : class template
    * used to initialize transformation via CRTP chain
    * created via inherited `TransformationBase::Base`_::transformation\_
    * creates `TransformationBase::Entry`_ instance and assignes it to the caller
    * assigns inputs, outputs, types functions, etc


Indirect access classes
"""""""""""""""""""""""

.. _Accessor:

Accessor : class
    * limited indirect access to `TransformationBase::Base`_ instance
    * access to `TransformationBase::Entry`_ by name or index via Handle_

.. _Args:

Args : struct
    * limited indirect access to `TransformationBase::Entry`_ instance
    * transformation input implementation
    * access to `TransformationBase::Entry`_'s data

.. _Atypes:

Atypes : struct
    * limited indirect access to `TransformationBase::Entry`_ instance
    * used for inputs' type checking
    * access to `TransformationBase::Entry`_'s ``DataType``

.. _Handle:

Handle : class
    * indirect access to `TransformationBase::Entry`_ instance
    * implements and redirects `TransformationBase::Entry`_ methods

.. _InputHandle:

InputHandle : class
    * limited indirect access to Source_
    * may be connected to OutputHandle_

.. _Rets:

Rets : struct
    * limited indirect access to `TransformationBase::Entry`_ instance
    * transformation output implementation
    * access to `TransformationBase::Entry`_'s data
    * may be (un)frozen

.. _Rtypes:

Rtypes : struct
    * limited indirect access to `TransformationBase::Entry`_ instance
    * used for outputs' type checking
    * access to `TransformationBase::Entry`_'s ``DataType``

.. _Sink:

Sink : struct
    * public indirect access to `TransformationBase::Entry`_ instance
    * named

.. _Source:

Source : struct
    * public indirect access to `TransformationBase::Entry`_ instance
    * may be connected to Sink_ instance
    * named

.. _OutputHandle:

OutputHandle : class
    * limited indirect access to Sink_
    * may be:
      + checked if depends on changeable_ instance

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

