#!/bin/bash
set -e
THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $THIS_DIR
cp lib/balance_*.c*   caffe/src/caffe/layers/
cp lib/balance_*layer.hpp  caffe/include/caffe/layers/
cd $THIS_DIR/caffe
if [ ! -e build ]; then
    mkdir build
fi
cd build
cmake .. -DUSE_CUDNN=ON -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
cd $THIS_DIR
echo "Done!"
