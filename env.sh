source ~/root-6.10.08/build/bin/thisroot.sh

export LD_LIBRARY_PATH=$PWD/build/:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD/pylib:$PYTHONPATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/build:$(pwd)/build/cuda

export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

