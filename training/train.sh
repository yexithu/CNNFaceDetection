#!/usr/bin/env sh
set -e

./caffe/build/tools/caffe train --solver=training/multistep_solver.prototxt $@
