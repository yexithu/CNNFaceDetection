#include "CaffePredictor.hpp"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <fstream>

using namespace std;

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
            << " deploy.prototxt network.caffemodel"
            << " root_of_test test.list result_to_save" << std::endl;
        return 1;
    }

    ::google::InitGoogleLogging(argv[0]);

    string model_file   = argv[1];
    string trained_file = argv[2];
    string test_root    = argv[3];
    string test_list    = argv[4];
    string result_file  = argv[5];
    CaffePredictor caffe_predictor(model_file, trained_file);

    /* Load labels. */
    vector<string> img_list;
    vector<int> label_list; 
    ifstream infile;
    infile.open(test_list.c_str(), ios::in);
    string line;
    char dir_buf[100];
    int label_buf;
    while(getline(infile, line)) {
        sscanf(line.c_str(), "%s%d", dir_buf, &label_buf);
        img_list.push_back(dir_buf);
        label_list.push_back(label_buf);
    }

    vector<float> result_list(img_list.size());
    random_shuffle(img_list.begin(), img_list.end());
    for (string img : img_list) {
        cv::Mat mat;
        mat = cv::imread(img, 0);
        cv::resize(mat, mat, cv::Size(25, 25));
        cv::imshow("Test", mat);
        vector<float> out = caffe_predictor.Predict(mat);
        cout << "Score" << out[1] << endl;
        cv::waitKey(0);
    }
}