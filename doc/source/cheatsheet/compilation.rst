Compilation
^^^^^^^^^^^

.. table::
   :widths: 100 80

   +-----------------------------------------+--------------------------------------------------------------------------+
   | ``mkdir build && cd build && cmake ..`` | Initial cmake configuration                                              |
   +-----------------------------------------+--------------------------------------------------------------------------+
   | ``cmake build``                         | Update cmake configuration for existing ``build`` folder                 |
   +-----------------------------------------+--------------------------------------------------------------------------+
   | ``cmake --build build``                 | Compile the project                                                      |
   +-----------------------------------------+--------------------------------------------------------------------------+
   | ``cmake --build build -- clean``        | Clean the project: all the arguments after ``--`` are passed to ``make`` |
   +-----------------------------------------+--------------------------------------------------------------------------+
   | ``cmake --build build -- -j4``          | Compile the project in parallel                                          |
   +-----------------------------------------+--------------------------------------------------------------------------+
   | ``cmake -DVARNAME=VALUE ..``            | Set a configuration variable during configuration of build.              |
   |                                         | Better do it during initial configuration.                               |
   |                                         | The list of useful variables is presented below.                         |
   +-----------------------------------------+--------------------------------------------------------------------------+
   | ``cmake -UVARNAME ..``                  | Unset a variable from cache                                              |
   +-----------------------------------------+--------------------------------------------------------------------------+



For ultimate cleaning up remove the entire content of ``build`` directory.
   
The list of useful configuration variables:


.. table::
   :widths: 100 80

   +----------------------+------------------------------------------+
   | Variable name        | Possible values and meaning              |
   +----------------------+------------------------------------------+
   | ``CMAKE_BUILD_TYPE`` | Release, Debug and other standard values |
   +----------------------+------------------------------------------+
   | ``NATIVE``           | ON, OFF.                                 |
   |                      | Use native set of CPU instructions       |
   +----------------------+------------------------------------------+
   | ``TRANS_DEBUG``      | ON, OFF.                                 |
   |                      | Show debug output of transformations     |
   +----------------------+------------------------------------------+
   | ``PARAM_DEBUG``      | ON, OFF.                                 |
   |                      | Show debug output for parameters         |
   +----------------------+------------------------------------------+
   | ``GRIDFILTER_DEBUG`` | ON, OFF.                                 |
   |                      | Show debug output for GridFilter class   |
   +----------------------+------------------------------------------+

