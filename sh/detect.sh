LD_LIBRARY_PATH=./caffe/distribute/lib/ \
./src/detect.bin \
   ./data/models/conv1_deploy.prototxt \
   ./data/models/conv1_iter_10000.caffemodel \
   ./data/testcase/1.png \
   ./data/testcase/conv1_10000.jpg
