#!/bin/bash
export CXX=/usr/bin/x86_64-linux-gnu-g++-5
export CC=/usr/bin/x86_64-linux-gnu-gcc-5
sudo apt-get -q -y install hdf5-tools

export CXXFLAGS=-D_GLIBCXX_USE_CXX11_ABI=0
cmake -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0 ..
