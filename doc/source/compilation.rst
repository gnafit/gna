Compilation
==============

You must have the following libraries installed prior to compilation:

* `Boost <http://www.boost.org/>`_, at least 1.40.0;
* `Eigen <http://eigen.tuxfamily.org/>`_ at least 3.2;
* `ROOT <http://root.cern.ch/>`_, at least 6.06 built with Python
  bindings support. The branch 6.08 is more preffered for now since ROOT is
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
packages can have different names)::

  $  sudo apt-get install libboost-all-dev hdf5-tools hdf5-helpers \
                          hdfview libeigen3-dev libgsl-dev


You also need Python 2.7, not very old `CMake
<http://www.cmake.org/>`_ and modern C++11 compiler (GCC or
clang). After all of that is installed, you can run the standard CMake
procedure::

  $ mkdir build
  $ cd build
  $ cmake ..
  $ cmake --build . -- {flags for underlying build engine here}

If the build was successfull, you can run the program::

  $ export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
  $ export PYTHONPATH=$PWD/../pylib:$PYTHONPATH
  $ cd ..
  $ python2 gna

If one wants to use the code at MACOS X, there can be problem with loading
dynamic library. Note that by default configuration of cmake with flag SHARED
for building the shared library the `.dylib` file will be produced. As far as
I understand ROOT can't load it directly, so the solution is to make a symlink
into like that::  

 $ ln -s $GNA_PATH/build/libGlobalNuAnalysis.dylib $ROOTSYS/lib/libGlobalNuAnalysis.so

If everything is fine, no output will be produced.
Incremental rebuild can be issued by::

  $ make -C build

If there are errors with unresolved symbols with C++ ABI tags during
linking or running, you may want to try to build with clang++. Remove
the build directory and set::

  $ export CXX=clang++

before starting the build procedure again. 

In case you have GCC 5.X compiler or newer
version you may use the following cmake command when generating the
build files with your additional flags if needed::

  $ export CXXFLAGS=-D_GLIBCXX_USE_CXX11_ABI=0
  $ cmake -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0 ..

Notice that if ABI mismatch encountered, the ROOT is also have to be recompiled
from scratch with compilation flags above -- ROOT branch 6.06 doesn't support GCC5 ABI, but ROOT 6.08 does.
