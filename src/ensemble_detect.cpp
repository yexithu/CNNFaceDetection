#include "EnsembleDetector.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <fstream>

using namespace std;

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
            << " deploy.prototxt network1.caffemodel network2.caffemodel"
            << " input.img output.image" << std::endl;
        return 1;
    }

    ::google::InitGoogleLogging(argv[0]);

    string model_file   = argv[1];
    string trained_file1 = argv[2];
    string trained_file2 = argv[3];
    string input    = argv[4];
    string output  = argv[5];

    EnsembleDetector ensemble_detector(model_file, trained_file1, trained_file2);
    cv::Mat img = cv::imread(input);
    ensemble_detector.Input(img);
    if (ensemble_detector.Detect()) {
        vector<Face> faces = ensemble_detector.GetFaces();
        cv::Mat out = ensemble_detector.GetOutput();
        for (auto f: faces) {
            cout << f.rect << "\t";
            if (f.gender) {
                cout << "Male ";
            } else {
                cout << "Female ";
            }
            if (f.is_smile) {
                cout << "Smileing "; 
            }
            cout << endl;
        }
        cv::imwrite(output, out);
    } else {
        cout << "Detection Fail." << endl;
    }
}