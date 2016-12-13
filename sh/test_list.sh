LD_LIBRARY_PATH=./caffe/distribute/lib/ \
./src/test_list.bin \
   ./models/gender_deploy.prototxt \
   ./data/models/conv2_gender_iter_6000.caffemodel \
   ./data/lists/gender_test.list \
   ./data/results/gender_result.txt
