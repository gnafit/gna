.. _NestedDict:

NestedDict
^^^^^^^^^^

Overview
""""""""

``NestedDict`` is a helper class very similar to the python ``dict`` or ``OrderedDict`` with the following extra features:

1. ``NestedDict`` enables attribute access as well as key access to the stored items:

   .. code-block:: python

       nd = NestedDict( a=1, b=2 )
       nd['a']
       # 1
       nd.a
       # 1

2. ``NestedDict`` enables nesting (unlimited):

   * Each stored dictionary will be converted to the ``NestedDict`` instance.
   * Assigning to ``nd['a.b']`` will create ``NestedDict`` ``nd['a']`` and assign ``nd['a']['b']``.
   * Assigning to ``nd.a.b`` also works, but only for existing ``nd.a``.
   * Nested ``NestedDict`` may be also  created by calling ``nd('a')`` or ``nd('a.b')`` or, etc.
   * Parent dictionary may be accessed by calling ``nd.parent()`` method.

3. Usual ``dict`` methods like ``set``, ``setdefault``, ``get`` and ``keys`` are also supported.

It should be noted that the keys ``set``, ``setdefault``, ``get`` and ``keys`` may be accessed only via ``[]`` operator
or the methods themselves, but not via attribute access.
