LD_LIBRARY_PATH=./caffe/distribute/lib/ \
./src/detect.bin \
   ./data/models/conv2_deploy.prototxt \
   ./data/models/conv2_bs1_iter_5000.caffemodel \
   ./data/testcase/1.png \
   ./data/testcase/conv2_bs1_5000_0.5_1.jpg
