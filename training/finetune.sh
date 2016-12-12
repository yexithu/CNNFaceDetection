#!/usr/bin/env sh
set -e

./caffe/build/tools/caffe train --solver=training/multistep_solver.prototxt --weights=data/models/conv2_bs_iter_4000.caffemodel $@
