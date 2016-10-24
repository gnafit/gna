Compilation
==============

You must have the following libraries installed prior to compilation:

* `Boost <http://www.boost.org/>`_, at least 1.40.0;
* `Eigen <http://eigen.tuxfamily.org/>`_ at least 3.2;
* `ROOT <http://root.cern.ch/>`_, at least 6.06 built with Python
  bindings support (tested with 6.06.06 and 6.06.08);
* `GSL <http://www.gnu.org/software/gsl/>`_
* `hdf5 <https://www.hdfgroup.org/HDF5/>`_, at least 1.8.14
* `h5py <http://www.h5py.org/>`_, at least 2.5.0
* `numpy <http://www.numpy.org/>`_, at least 1.10
* `scipy <http://www.scipy.org/>`_
* `matplotlib <http://matplotlib.org/>`_
* `PyYAML <http://pyyaml.org/>`_
* `IPython <http://ipython.org/>`_

In case of Debian-like Linux distribution to install binary packages you can
use the following command (tested in Ubuntu 16.10, depending on your distro
packages can have different names)::

  $  sudo apt-get install libboost-all-dev hdf5-tools hdf5-helpers hdfview libeigen3-dev libgsl-dev

You may install Python packages manually through distro package manager or by using `requirements.txt` file
that is located in the project root::

  $ pip -r install requirements.txt 

You also need Python 2.7, not very old `CMake
<http://www.cmake.org/>`_ and modern C++11 compiler (GCC or
clang). After all of that is installed, you can run the standard CMake
procedure::

  $ mkdir build
  $ cd build
  $ cmake ..
  $ make

If the build was successfull, you can run the program::

  $ export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
  $ export PYTHONPATH=$PWD/../pylib:$PYTHONPATH
  $ cd ..
  $ python2 gna

If everything went fine, no output will be produced. Incremental
rebuild can be issued by::

  $ make -C build

If there are errors about unresolved symbols with C++ ABI tags during
linking or running, you may want to try to build with clang++. Remove
the build directory and set::

  $ export CXX=clang++

before starting the build procedure again. 

In case you have GCC 5.X compiler
version you may use the following cmake command when generating the
build files with your additional flags if needed::
  $ export CXXFLAGS=-D_GLIBCXX_USE_CXX11_ABI=0
  $ cmake -DCMAKE_CXXFLAGS=-D_GLIBCXX_USE_CXX11_ABI=0 ..
Notice that if ABI mismatch encountered the ROOT is also have to be recompiled
from scratch with compilation flags above (current versions of ROOT do not support CXX11 ABI, but hopefully it's coming in future versions). As rule of thumbs I would suggest to use that flags from the  beginning.
