LD_LIBRARY_PATH=./caffe/distribute/lib/ \
./src/detect.bin \
   ./data/models/conv2_deploy.prototxt \
   ./data/models/conv2_bs_iter_10000.caffemodel \
   ./data/testcase/1.png \
   ./data/testcase/conv2_bs_10000_0.8.jpg
