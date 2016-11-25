#include "Detector.hpp"
#include "CaffePredictor.hpp"

using namespace caffe;
using std::string;
using std::vector;
Detector::Detector(const string& model_file, const string& trained_file)
	: FACESIZE(25), HALFSIZE(12), SCALERATE(1.25), STRIDE(3) {
	predictor_.reset(new CaffePredictor(model_file, trained_file));
}

void Detector::Input(const cv::Mat img) {
	std::cout << "Detector::Input" << std::endl;
	input_ = img.clone();
	cv::resize(input_, input_, cv::Size(), 0.5, 0.5);
	cv::cvtColor(input_, gray_, CV_BGR2GRAY);
	output_ = input_.clone();
	faces_.clear();
}

bool Detector::Detect() {
	std::cout << "Detector::Detect" << std::endl;
	vector<cv::Point> centers;
	vector<vector<float> > score_map;
	cv::imshow("Gray", gray_);
	cv::Size size = gray_.size();
	for (int i = 0; i < gray_.rows; i+= STRIDE) {
		vector<float> row_score_map;
		for (int j = 0; j < gray_.cols; j += STRIDE) {
			cv::Point p(j, i);
			cv::Rect rect = _roi(p);
			if (!_check(rect, size)) {
				continue;
			}

			centers.push_back(p);
			cv::Mat patch = gray_(rect);
			// cv::imshow("Patch", patch);
			// cv::waitKey(30);
			float score = predictor_->Predict(patch)[1];
			// std::cout << "Score " << score << std::endl;
			row_score_map.push_back(score);

			if (score > 0.5) {
				// cv::waitKey();
				cv::rectangle(output_, rect, cv::Scalar(0, 0, 255));
			}
		}
		score_map.push_back(row_score_map);
	}
	cv::imshow("Output", output_);
	cv::waitKey();
	return true;
}

vector<cv::Rect> Detector::GetFaces() {
	std::cout << "Detector::GetFaces" << std::endl;
	return vector<cv::Rect>();
}

cv::Mat Detector::GetOutput() {
	std::cout << "Detector::GetOutput" << std::endl;
	return output_;
}
