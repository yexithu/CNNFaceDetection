#!/usr/bin/env sh
set -e

./caffe/build/tools/caffe train --solver=training/multistep_solver.prototxt --snapshot=./data/models/conv2_bs_iter_2000.solverstate $@
