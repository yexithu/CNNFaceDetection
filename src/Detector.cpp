#include "Detector.hpp"
#include "CaffePredictor.hpp"
#include <cmath>

using namespace caffe;
using std::string;
using std::vector;
using std::floor;
using std::round;
Detector::Detector(const string& model_file, const string& trained_file)
	: FACESIZE(25), HALFSIZE(12), SCALERATE(1.5), STRIDE(3), GROUPTHRESHOLD(1) {
	predictor_.reset(new CaffePredictor(model_file, trained_file));
}

void Detector::Input(const cv::Mat img) {
	std::cout << "Detector::Input" << std::endl;
	input_ = img.clone();
	// cv::resize(input_, input_, cv::Size(), 0.3, 0.3);
	cv::cvtColor(input_, gray_, CV_BGR2GRAY);
	output_ = input_.clone();
	faces_.clear();
}

void Detector::AddPadding(cv::Rect& r, int padding){

	r.x -= padding;
	r.width += 2 * padding;
	r.y -= padding;
	r.height += 2 * padding;

	r.x = r.x>=0? r.x: 0;
	r.y = r.y>=0? r.y: 0;
}
void Detector::AppendRectangles(
    vector<cv::Rect>& old_list,
    vector<cv::Rect>& new_list) {

    vector<cv::Rect> rects_to_be_appended;
    for(auto new_rect : new_list) {
        bool flag = false;
        for(auto old_rect : old_list) {

			auto padding = old_rect.width / 4;

			auto larger_new = new_rect;
			auto smaller_new = new_rect;
			AddPadding(larger_new, padding);
			AddPadding(smaller_new, - padding);

			bool new_covers_old = cover(new_rect, old_rect);
			bool smaller_new_overlaps_old = overlap(smaller_new, old_rect);
			bool larger_new_covers_old = cover(larger_new, old_rect);

            if(new_covers_old || smaller_new_overlaps_old || larger_new_covers_old) {
                flag = true;
                break;
            }
        }
        if(!flag) {
            rects_to_be_appended.push_back(new_rect);
        }
    }

    for(auto i : rects_to_be_appended) {
        old_list.push_back(i);
    }
    return;
}


bool Detector::Detect() {
	float rate = 1;
	cv::Mat img = gray_.clone();
	while(true) {
		cv::Size size = img.size();
		if (size.width < FACESIZE || size.height< FACESIZE)
			break;

		vector<cv::Rect> rects = ScanImage(img);
		for (cv::Rect &rect: rects) {
			rect.x = round(rect.x * rate);
			rect.y = round(rect.y * rate);
			rect.width = round(rect.width * rate);
			rect.height = round(rect.height * rate);
		}
		AppendRectangles(faces_, rects);
		// for (auto r: rects)
		// 	faces_.push_back(r);
		rate *= SCALERATE;
		cv::resize(gray_, img, cv::Size(), 1 / rate, 1 / rate);
	}

	cv::Size in_size = input_.size();
	in_size.width += 5;
	in_size.height += 5;
	cv::resize(output_, output_, in_size);
	for (cv::Rect rect: faces_) {
		cv::rectangle(output_, rect, cv::Scalar(0, 0, 255));
	}
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

vector<cv::Rect> Detector::ScanImage(cv::Mat &img) {
	std::cout << "Detector::ScanImage" << std::endl;
	cv::Size size = img.size();
	vector<cv::Rect>  rects;
	for (int i = 0; i < img.rows; i+= STRIDE) {
		// vector<float> row_score_map;
		for (int j = 0; j < img.cols; j += STRIDE) {
			cv::Point p(j, i);
			cv::Rect rect = _roi(p);
			if (!_check(rect, size)) {
				continue;
			}
			cv::Mat patch = img(rect);
			float score = predictor_->Predict(patch)[1];
			if (score > 0.5) {
				//std::cout << score << std::endl;
				//cv::imshow("Patch", patch);
				//cv::waitKey();
 				rects.push_back(rect);
			}
		}
	}
	groupRectangles(rects, GROUPTHRESHOLD);
	return rects;
}
