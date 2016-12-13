LD_LIBRARY_PATH=./caffe/distribute/lib/ \
./src/ensemble_detect.bin \
   ./models/face.prototxt \
   ./models/face1.model \
   ./models/face2.model \
   ./data/testcase/3.png \
   ./data/testcase/ensemble/3.jpg
