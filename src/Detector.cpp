#include "Detector.hpp"
#include "CaffePredictor.hpp"
#include <cmath>

using namespace caffe;
using std::string;
using std::vector;
using std::floor;
using std::round;
Detector::Detector(const string& model_file, const string& trained_file)
	: FACESIZE(25), HALFSIZE(12), SCALERATE(1.25), STRIDE(3), GROUPTHRESHOLD(5) {
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

// bool Detector::Detect() {
// 	std::cout << "Detector::Detect" << std::endl;
// 	vector<cv::Point> centers;
// 	vector<vector<float> > score_map;
// 	cv::imshow("Gray", gray_);
// 	cv::Size size = gray_.size();


// 	for (int i = 0; i < gray_.rows; i+= STRIDE) {
// 		vector<float> row_score_map;
// 		for (int j = 0; j < gray_.cols; j += STRIDE) {
// 			cv::Point p(j, i);
// 			cv::Rect rect = _roi(p);
// 			if (!_check(rect, size)) {
// 				continue;
// 			}

// 			centers.push_back(p);
// 			cv::Mat patch = gray_(rect);
// 			// cv::imshow("Patch", patch);
// 			// cv::waitKey(30);
// 			float score = predictor_->Predict(patch)[1];
// 			// std::cout << "Score " << score << std::endl;
// 			row_score_map.push_back(score);

// 			if (score > 0.5) {
// 				// cv::waitKey();
//  				faces_.push_back(rect);
// 				// cv::waitKey(100);
// 				cv::rectangle(output_, rect, cv::Scalar(0, 0, 255));
// 			}
// 		}
// 		score_map.push_back(row_score_map);
// 	}
// 	groupRectangles(faces_, 3, 0.1);
// 	cv::Mat compare = input_.clone();
// 	for (cv::Rect rect: faces_) {
// 		cv::rectangle(compare, rect, cv::Scalar(0, 0, 255));
// 	}
// 	cv::imwrite("data/testcase/3.jpg", compare);
// 	// cv::imshow("Output", output_);
// 	// cv::waitKey();
// 	return true;
// }


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
		rate *= SCALERATE;
		cv::resize(img, img, cv::Size(), 1 / rate, 1 / rate);
	}

	cv::Size in_size = input_.size();
	in_size.width += 5;
	in_size.height += 5;
	cv::resize(output_, output_, in_size);
	for (cv::Rect rect: faces_) {
		cv::rectangle(output_, rect, cv::Scalar(0, 0, 255));
	}
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
 				rects.push_back(rect);
			}
		}
	}
	groupRectangles(rects, GROUPTHRESHOLD);
	return rects;
}
