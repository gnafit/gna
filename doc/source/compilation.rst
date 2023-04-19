Installation
============

Requirements
------------

You must have the following libraries installed prior to compilation:

* `Boost <http://www.boost.org/>`_, at least 1.40.0;
* `Eigen <http://eigen.tuxfamily.org/>`_ at least 3.2;
* `CMake <http://www.cmake.org/>`_  at least 3.5;
* `ROOT <http://root.cern.ch/>`_, at least 6.06 built with Python3
  bindings support and C++17 standard enabled. The branch 6.08 is more preferred for now since ROOT is
  able to generate bindings following GCC5 ABI.
* `GSL <http://www.gnu.org/software/gsl/>`_
* `hdf5 <https://www.hdfgroup.org/HDF5/>`_, at least 1.8.14
* `h5py <http://www.h5py.org/>`_, at least 2.5.0
* `numpy <http://www.numpy.org/>`_, at least 1.10
* `scipy <http://www.scipy.org/>`_
* `matplotlib <http://matplotlib.org/>`_
* `PyYAML <http://pyyaml.org/>`_
* `IPython <http://ipython.org/>`_

In case of Debian-like Linux distribution to install binary packages you can
use the following command (tested in Ubuntu 16.04, depending on your distro
packages can have different names):

.. code-block:: bash

    sudo apt-get install libboost-all-dev hdf5-tools hdf5-helpers \
                          hdfview libeigen3-dev libgsl-dev cmake

In Fedora 26:

.. code-block:: bash

    sudo yum install boost-devel hdf5-devel hdfview eigen3-devel \
                     gsl-devel

In Arch:

.. code-block:: bash

    sudo pacman -S boost hdf5 eigen gsl cmake

You also need Python 2.7 and modern C++17-compatible compiler (GCC or clang).

Python requirements
-------------------

The list of python modules, required fro GNA is listed in :download:`requirements.txt <../../requirements.txt>`:

.. literalinclude:: ../../requirements.txt

They may be installed via system package manager, or manually via pip only for
current user:

.. code-block:: bash

    pip install --user -r requirements.txt

`pip` will install only the packages, that are missing in the system.

Setting ROOT environment
------------------------

The ROOT environmental variables should be properly set. The necessary source files are provided within ROOT installation
for bash/tcsh/fish [#]_. Assuming root is installed in `/home/user/path/to/root/`, one may use:

.. code-block:: bash

    source /home/user/path/to/root/bin/thisroot.sh

or

.. code-block:: bash

    source /home/user/path/to/root/bin/thisroot.fish

or

.. code-block:: bash

    source /home/user/path/to/root/bin/thisroot.csh

depending on the shell of the choice.


Compilation
-----------

After dependencies are installed, one should follow the standard CMake procedure:

.. code-block:: bash

   mkdir build
   cd build
   cmake ..
   cmake --build . -- {flags for underlying build engine here}

For available options for configuration and build refer to :doc:`Build options <cheatsheet/build_options>`.
 
In order to use GNA, the following environmental should be set:

.. code-block:: bash
    
   export LD_LIBRARY_PATH=path_to_gna_root/build:$LD_LIBRARY_PATH
   export PYTHONPATH=path_to_gna_root/pylib:$PYTHONPATH

`$LD_LIBRARY_PATH` contains pathes where shared libraries are looked up for
loading at runtime.
`$PYTHONPATH` contains pathes where Python interpreter looks for modules at
runtime.

If the build was sucessfull, you can dry-run the program to check that
it is working properly:

.. code-block:: bash

   python gna

Bulding on MacOS
----------------

If one wants to use the code at MACOS X, there can be problem with loading
dynamic library. Note that by default configuration of cmake with flag SHARED
for building the shared library the `.dylib` file will be produced. As far as
I understand ROOT can't load it directly, so the solution is to make a symlink
into like that:

.. code-block:: bash

  ln -s $GNA_PATH/build/libGlobalNuAnalysis.dylib $ROOTSYS/lib/libGlobalNuAnalysis.so

If everything is fine, no output will be produced.
Incremental rebuild can be issued one of the commands:

.. code-block:: bash

   make -C build
   cmake --build build


Troubleshooting ABI mismatch with old ROOT versions
---------------------------------------------------

If you are using old ROOT versions (below 6.08) it is possible to get an ABI
mismatch errors when compiling with GCC 5.X or newer. Those errors are caused
by usage of old ABI in ROOT itself. 
Consider using the following macro to force compiler to use older ABI

.. code-block:: bash

   export CXXFLAGS=-D_GLIBCXX_USE_CXX11_ABI=0
   cmake -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0 ..

Notice that in such case, ROOT also have to be recompiled
from scratch with compilation flags above.


CUDA support
------------

GNA supports particulary porting of computations to GPGPU. `CUDA-enable NVIDIA GPU <https://developer.nvidia.com/cuda-gpus>`_ is necessary to use this option. To enable CUDA support in GNA NVIDIA Driver is have to be installed (v384 is tested).

The following software have to be installed additionally:

* `CUDA Toolkit <https://developer.nvidia.com/cuda-downloads>`_, at least 7.5
* GCC 5.x, G++ 5.x (not higher)

To enable CUDA support in GNA some variables have to be set:

.. code-block:: bash

    cmake -DSINGLE_PRECISION=ON -DCUDA_SUPPORT=1 -DCUDA_DEBUG_INFO=0 ..


Setting of the threshold value can be skipped as it has default value.

.. [#] fish support was introduced recently, around ROOT 6.18
