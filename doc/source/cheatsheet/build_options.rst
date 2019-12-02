Build options
^^^^^^^^^^^^^

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

   +------------------------+---------------------------------------------------------------------+
   | Variable name          | Possible values and meaning                                         |
   +------------------------+---------------------------------------------------------------------+
   | ``CMAKE_BUILD_TYPE``   | Release, Debug and other standard values                            |
   +------------------------+---------------------------------------------------------------------+
   | ``CMAKE_CXX_STANDARD`` | 14, 17.                                                             |
   |                        | Version of C++ standard to use                                      |
   +------------------------+---------------------------------------------------------------------+
   | ``CMAKE_C_COMPILER``   | Which C compiler to use: gcc, clang, gcc-9 or other.                |
   +------------------------+---------------------------------------------------------------------+
   | ``CMAKE_CXX_COMPILER`` | Which C++ compiler to use: g++, clang++, g++9 or other.             |
   +------------------------+---------------------------------------------------------------------+
   | ``LTO``                | ON, OFF.                                                            |
   |                        | Enable link-time optimizations                                      |
   +------------------------+---------------------------------------------------------------------+
   | ``SINGLE_PRECISION``   | ON, OFF.                                                            |
   |                        | Enable optional usage of single precision floats in transformations |
   +------------------------+---------------------------------------------------------------------+
   | ``CUDA_SUPPORT``       | 1, 0.                                                               |
   |                        | Enable CUDA support for computations on GPU                         |
   +------------------------+---------------------------------------------------------------------+
   | ``CUDA_DEBUG_INFO``    | 1, 0.                                                               |
   |                        | Enable debug logging for CUDA                                       |
   +------------------------+---------------------------------------------------------------------+
   | ``UB_SANITIZE``        | ON, OFF.                                                            |
   |                        | Use undefined behaviour sanitizer if                                |
   |                        | supported by the compiler                                           |
   +------------------------+---------------------------------------------------------------------+
   | ``GENERATE_PROFILE``   | ON, OFF.                                                            |
   |                        | Allow generation of execution profile for                           |
   |                        | profile-guided oprimizations                                        |
   +------------------------+---------------------------------------------------------------------+
   | ``USE_PROFILE``        | ON, OFF.                                                            |
   |                        | Use generated profile of execution for                              |
   |                        | profile-guided oprimizations at compile-time                        |
   +------------------------+---------------------------------------------------------------------+
   | ``NATIVE``             | ON, OFF.                                                            |
   |                        | Use native set of CPU instructions                                  |
   +------------------------+---------------------------------------------------------------------+
   | ``TRANS_DEBUG``        | ON, OFF.                                                            |
   |                        | Show debug output of transformations                                |
   +------------------------+---------------------------------------------------------------------+
   | ``PARAM_DEBUG``        | ON, OFF.                                                            |
   |                        | Show debug output for parameters                                    |
   +------------------------+---------------------------------------------------------------------+
   | ``GRIDFILTER_DEBUG``   | ON, OFF.                                                            |
   |                        | Show debug output for GridFilter class                              |
   +------------------------+---------------------------------------------------------------------+

