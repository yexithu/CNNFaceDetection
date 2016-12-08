LD_LIBRARY_PATH=./caffe/distribute/lib/ \
./src/detect.bin \
   ./data/models/conv2_deploy.prototxt \
   ./data/models/conv2_bs_iter_4000.caffemodel \
   ./data/testcase/1.png \
   ./data/testcase/conv2_bs_4000_0.5.jpg
