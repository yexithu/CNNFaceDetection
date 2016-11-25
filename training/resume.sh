#!/usr/bin/env sh
set -e

./caffe/build/tools/caffe train --solver=training/facenet_solver.prototxt --snapshot=./data/models/conv2_iter_377.solverstate $@
