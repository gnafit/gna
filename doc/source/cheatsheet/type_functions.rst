Type functions
~~~~~~~~~~~~~~

Description
^^^^^^^^^^^

In GNA there is a number of predefined types functions for setting
sizes(shapes) of
outputs based on inputs sizes, see `core/TransformationBase.hh`:

1. ``Atypes::pass<I,J>`` - assigns shape of `I`-th inputs to `J`-th output.
2. ``Atypes::passAll`` - assigns shape of each input to corresponding output. If
   the number of inputs and outputs is not the same, exception will be thrown.
   In case of single input and mltiple outputs assign its' size to each
   output.
3. ``Atypes::ifSame`` - checks that all inputs are of the same type (shape and
   content description).
4. ``Atypes::ifSameShape`` - checks that all inputs have the same shape.
