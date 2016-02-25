Compilation
==============

You must have the following libraries installed prior to compilation:

* `Boost <http://www.boost.org/>`_, at least 1.40.0;
* `Eigen <http://eigen.tuxfamily.org/>`_ at least 3.2;
* `ROOT <http://root.cern.ch/>`_, at least 6.06 built with Python
  bindings support;
* `GSL <http://www.gnu.org/software/gsl/>`_
* `hdf5 <https://www.hdfgroup.org/HDF5/>`_, at least 1.8.14
* `h5py <http://www.h5py.org/>`_, at least 2.5.0
* `numpy <http://www.numpy.org/>`_, at least 1.10
* `scipy <http://www.scipy.org/>`_
* `matplotlib <http://matplotlib.org/>`_
* `PyYAML <http://pyyaml.org/>`_
* `IPython <http://ipython.org/>`_

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

If there are errors about unresolved symbols with c++ abi tags during
linking or running, you may want to try to build with clang++. Remove
the build directory and set::

  $ export CXX=clang++

before starting the build procedure again.
