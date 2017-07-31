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
    calculation unit with arbitrary number of inputs and at least one output

Input, argument, source
    calculation node input data

Output, return value, sink
    calculation node output data

Connection, connect
    procedure of the connection between one `TransformationBase::Entry`_'s output to the other `TransformationBase::Entry`_'s input

GNAObject header
^^^^^^^^^^^^^^^^

.. _GNAObject:

GNAObject : class
    * Carries lists of:
      + variable_ isntances via VariableDescriptor_
      + evaluable_ isntances via EvaluableDescriptor_
      + transformation instances via TransformationDescriptor_
    * Inherits `TransformationBase::Base`_

      + carries list of `TransformationBase::Entry`_ instances

      + has transformation_ member function used to initialize transformations

    * Inherits `ParametrizedTypes::Base`_

      + carries list of `ParametrizedTypes::Entry`_ instances

      + has variable\_ member function used to define variables

.. _GNASingleObject:

GNASingleObject : class
    * Implements some shorcuts for GNAObject_ with only one output
    * Inherits GNAObject_ and SingleOutput_

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

.. _inconstant_header:

inconstant_header : struct
    * carries information to track dependencies:
        + taintflag_
        + list of taint signal emitters
        + list of taint signal observers
    * base for inconstant_data_

.. _references:

references : struct
    * a container for changeable_ instances
    * used in inconstant_header_
    * why manually allocated? TBD

Variables
"""""""""

Variables represent simple data types (double) with taintflag_ propagation feature.
Variables within same namespace (python side) are distinguished by their name.
Creating several variable_ or parameter_ instances with the same name will manage the same data.

Variables are mostly used on C++ side. On python side the Parameter, Uncertain, etc are used as accessors.

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
    * usuall created via `mkdep()`_ function

.. _evaluable:

evaluable : class template
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
    * shares it's taintflag_ with all the entries

.. _`ParametrizedTypes::Entry`:

Entry : class
    * a class to access variable's value

    * contains pointers to:

      + parameter_ par — the parameter

      + variable_ var — pointer to par (of the base class)

      + variable_ field

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

..    * may rebind MemFunction_ instances to `TransformationBase::Entry`_ instances accordingly
      * may rebind MemTypesFunction_ instances to `TransformationBase::Entry`_ instances accordingly


TransformationTypes namespace (TransformationBase)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Functions
"""""""""

.. _Function:

Function : std::function
    * (Args_, Rets_)
    * an implementation of the particular transformation

.. _TypesFunction:

TypesFunction : std::function
    * (Atypes_, Rtypes_)
    * an transformation input/output types initialization and checking

.. _MemFunction:

MemFunction : std::function
    * template
    * (T* this, Args_, Rets_)
    * an implementation of the particular transformation
    * requires the object to be passed as the first argument (needs binding)

.. _MemTypesFunction:

MemTypesFunction : std::function
    * template
    * (T*, Atypes_, Rtypes_)
    * an transformation input/output types initialization and checking
    * requires the object to be passed as the first argument (needs binding)

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
    * base class for the calculation node representation
    * has methods to:

      + add sources/sinks

      + evaluate/update types/values

      + freeze/unfreeze/touch

    * gives access to:

      + sources/sinks

      + data

      + taintflag

    * accessed via Handle_ class
    * named

.. _Initializer:

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
    * Base class for type errors
    * Just pass a message to ``std::runtime_error`` constructor

.. _CalculationError:

CalculationError : class
    * Can be throwed if transformation cannot be computed: invalid source and
      etc..
    * Appears only in ``operator[](int i)`` for ``Args, Rets`` and in
      ``Entry::data(int i)``


.. _SinkTypeError:

SinkTypeError : class
    * Inherits from TypeError
    * Throwed when type function fails on constructing sink via
      ``rets.error(message)``

.. _SourceTypeError:

SourceTypeError : class
    * Inherits from TypeError
    * Throwed when type function fails on constructing source via
      ``args.error(message)``

.. _`Transformation header`:

Transformation header
^^^^^^^^^^^^^^^^^^^^^

.. _InputDescriptor:

InputDescriptor : class
    * a wrapper to the InputHandle_
    * implements various forms of the `connect()` function

.. _OutputDescriptor:

OutputDescriptor : class
    * a wrapper to the OutputHandle_

.. _TransformationDescriptor:

TransformationDescriptor : class
    * a wrapper to the `TransformationBase::Entry`_
    * carries also lists of InputDescriptor_ instances and OutputDescriptor_ instances

UncertainParameter header
^^^^^^^^^^^^^^^^^^^^^^^^^

The header contains variaous variable_ and parameter_ views, defined as transformations
and used on python side.

.. _GaussianParameter:

GaussianParameter : class template
    * a nickname for `Parameter (Uncertain)`_
    * represents normally distributed variable with central value and sigma

.. _`Parameter (Uncertain)`:

Parameter : class template
    * derives _Uncertain
    * carries parameter_ instance for the variable_, i.e. may set it's value
    * may:
      + set parameter_'s value
      + set parameter_'s value in terms of sigma relative to it's central position
      + define limits (used for minimization)
    * the class is used as an input for the minimization

.. _ParameterWrapper:

ParameterWrapper : class template
    * a simple wrapper for the parameter_ class meant to use on python side
    * has set and get methods

.. _Uncertain:

Uncertain : class template
    * GNAObject_ represending a transformation with no inputs and one output
    * output is connected with variable_ instance (connection is name based)
    * carries also information about variable_'s central value and uncertainty (sigma)

.. _UniformAngleParameter:

UniformAngleParameter : class template
    * derives Parameter_
    * represents an angle in radiance defined in :math:`[-\pi, \pi)`


ParametricLazy.hpp header
^^^^^^^^^^^^^^^^^^^^^^^^^

Defines code for the evaluable_ creation based on math expressions.

.. _`mkdep()`:

Defines `mkdep()` function which does the job.

No additional reference for now (it's magic).




