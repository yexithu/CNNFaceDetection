#!/bin/bash -l
#$ -S /bin/bash
#$ -N $2

LD_LIBRARY_PATH=./caffe/distribute/lib/ \
./src/ensemble_detect.bin \
   ./models/face.prototxt \
   ./models/face1.model \
   ./models/face2.model \
   .$1 \
   .$2
