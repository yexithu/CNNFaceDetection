#include "Detector.hpp"
#include <opencv2/opencv.hpp>
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
            << " input.img output.image" << std::endl;
        return 1;
    }

    ::google::InitGoogleLogging(argv[0]);

    string model_file   = argv[1];
    string trained_file = argv[2];
    string input    = argv[3];
    string output  = argv[4];

    Detector detector(model_file, trained_file);
    cv::Mat img = cv::imread(input);
    detector.Input(img);
    if (detector.Detect()) {
        vector<cv::Rect> rects = detector.GetFaces();
        cv::Mat out = detector.GetOutput();
        cv::imwrite(output, out);
    } else {
        cout << "Detection Fail." << endl;
    }

}