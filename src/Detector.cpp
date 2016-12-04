#include <omp.h>
#include "Detector.hpp"
#include "CaffePredictor.hpp"
#include <cmath>

using namespace caffe;
using std::string;
using std::vector;
using std::floor;
using std::round;

#define THREADNUM 4
Detector::Detector(const string& model_file, const string& trained_file)
	: FACESIZE(25), HALFSIZE(12), SCALERATE(1.5), STRIDE(3), GROUPTHRESHOLD(5) {
	predictor_.reset(new CaffePredictor(model_file, trained_file));

	omp_set_num_threads(THREADNUM);
	multi_predictors_.resize(THREADNUM);
	for (int i = 0; i < THREADNUM; ++i) {
		multi_predictors_[i].reset(new CaffePredictor(model_file, trained_file));
	}
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

void Detector::ParrelTest(cv::Rect rect) {
	int id = omp_get_thread_num();
	// printf("%d\n", id);
	// cv::imshow("Window", multi_grays_[id]);
	// cv::waitKey();
	cv::Mat patch = multi_grays_[id](rect);
	if (multi_predictors_[id]->Predict(patch)[1] > 0.5) {
		multi_rects_[id].push_back(rect);
	}
	// printf("End %d\n", id);
}
vector<cv::Rect> Detector::ScanImage(cv::Mat &img) {
	std::cout << "Detector::ScanImage" << std::endl;

	multi_grays_.clear();
	multi_grays_.resize(THREADNUM);
	multi_rects_.clear();
	multi_rects_.resize(THREADNUM);

	for (int i = 0; i < THREADNUM; ++i) {
		multi_grays_[i] = img.clone();
	}
	cv::Size size = img.size();
	vector<cv::Rect>  rects;
	vector<cv::Rect>  allRects;
	for (int i = 0; i < img.rows; i+= STRIDE) {
		// vector<float> row_score_map;
		for (int j = 0; j < img.cols; j += STRIDE) {
			cv::Point p(j, i);
			cv::Rect rect = _roi(p);
			if (!_check(rect, size)) {
				continue;
			}
			allRects.push_back(rect);
		}
	}

	#pragma omp parallel for
	for (int i = 0; i < allRects.size(); ++i) {
		ParrelTest(allRects[i]);
	}

	int total_size = 0;
	for (int i = 0; i < multi_rects_.size(); ++i) {
		total_size += multi_rects_[i].size();
	}

	rects.reserve(total_size);
	for (int i = 0; i < multi_rects_.size(); ++i) {
		rects.insert(rects.end(), multi_rects_[i].begin(), multi_rects_[i].end());
	}
	groupRectangles(rects, GROUPTHRESHOLD);
	return rects;
}

