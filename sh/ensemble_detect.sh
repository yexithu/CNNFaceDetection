LD_LIBRARY_PATH=./caffe/distribute/lib/ \
./src/detect.bin \
   ./data/models/conv2_deploy.prototxt \
   ./data/models/conv2_face_bs1_iter_6000.caffemodel \
   ./data/models/conv2_face_bs1_iter_6000.caffemodel \
   ./data/testcase/1.png \
   ./data/testcase/conv2_face_bs1_6000_0.5_1.jpg
