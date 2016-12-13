#!/usr/bin/env sh
set -e

./caffe/build/tools/caffe train --solver=training/gender_solver.prototxt --weights=models/conv2_face1.caffemodel $@
