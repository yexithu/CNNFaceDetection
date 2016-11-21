#!/usr/bin/env sh
set -e

./caffe/bin/caffe.bin train --solver=training/facenet_solver.prototxt $@
