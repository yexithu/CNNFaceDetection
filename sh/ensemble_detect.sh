LD_LIBRARY_PATH=./caffe/distribute/lib/ \
./src/ensemble_detect.bin \
   ./data/models/conv2_deploy.prototxt \
   ./data/models/105conv2_bs11_iter_4000.caffemodel \
   ./data/models/conv2_bs11_iter_40000.caffemodel \
   ./data/testcase/5.jpg \
   ./data/testcase/ensemble/5.jpg
