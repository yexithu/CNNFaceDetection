LD_LIBRARY_PATH=./caffe/distribute/lib/ \
./src/detect.bin \
   ./data/models/conv2_deploy.prototxt \
   ./data/models/conv2_naive.caffemodel \
   ./data/testcase/1.jpg \
   ./data/testcase/2.jpg
