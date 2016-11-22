LD_LIBRARY_PATH=./caffe/distribute/lib/ \
./src/test_list.bin \
   ./data/models/conv2_deploy.prototxt \
   ./data/models/conv2_naive.caffemodel \
   ./data/lists/naive_test.list \
   ./data/results/naive_result.txt
