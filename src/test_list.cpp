#include "CaffePredictor.hpp"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <fstream>

using namespace std;

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
            << " deploy.prototxt network.caffemodel"
            << " test.list result_to_save" << std::endl;
        return 1;
    }

    ::google::InitGoogleLogging(argv[0]);

    string model_file   = argv[1];
    string trained_file = argv[2];
    string test_list    = argv[3];
    string result_file  = argv[4];
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
    infile.close();

    vector<float> result_list;
    // random_shuffle(img_list.begin(), img_list.end());
    int count = 0;
    for (string img : img_list) {
        if (count % 100 == 0) {
            cout << "Predicting " << count << endl;
        }
        count += 1;
        cv::Mat mat;
        mat = cv::imread(img, 0);
        cv::resize(mat, mat, cv::Size(25, 25));
        // cv::imshow("Test", mat);
        vector<float> out = caffe_predictor.Predict(mat);
        // cout << "Score" << out[1] << endl;
        // cv::waitKey(0);        
        result_list.push_back(out[1]);
    }

    ofstream outfile;
    outfile.open(result_file.c_str(), ios::out);
    int correct_count = 0;
    for (size_t i = 0; i < img_list.size(); ++i) {
        outfile << img_list[i] << " " << label_list[i] << " " << result_list[i] << endl;
        if (result_list[i] > 0.5) {
            if (label_list[i]) {
                correct_count += 1;
            }
        }
        else {
            if (!label_list[i]) {
                correct_count += 1;   
            }
        }
    }
    outfile.close();
    std::cout << "Acc: " << correct_count * 1.f / result_list.size() << std::endl;
}