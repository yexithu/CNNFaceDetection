#include "EnsembleDetector.hpp"
#include "CaffePredictor.hpp"
#include "rectangles.hpp"
#include <cmath>
#include <algorithm>

using namespace caffe;
using std::string;
using std::vector;
using std::floor;
using std::round;
using std::min;

const string GENDER_MODEL = "./models/gender.prototxt";
const string GENDER_TRAINED = "./models/gender.model";
const float GENDER_THRESHOLD = 0.5;

const string SMILE_MODEL = "./models/smile.prototxt";
const string SMILE_TRAINED = "./models/smile.model";
const float SMILE_THRESHOLD = 0.8;

EnsembleDetector::EnsembleDetector(const string& model_file,
						   const string& trained_file1,
						   const string& trained_file2)
	: FACESIZE(25), HALFSIZE(12), SCALERATE(1.5), STRIDE(3), GROUPTHRESHOLD(1), SCORETHRESHOLD(0.7), EPS(0.2) {

	predictor1_.reset(new CaffePredictor(model_file, trained_file1));
	predictor2_.reset(new CaffePredictor(model_file, trained_file2));

	predictor_gender_.reset(new CaffePredictor(GENDER_MODEL, GENDER_TRAINED));
	predictor_smile_.reset(new CaffePredictor(SMILE_MODEL, SMILE_TRAINED));
}

void EnsembleDetector::Input(const cv::Mat img) {
	std::cout << "EnsembleDetector::Input" << std::endl;
	input_ = img.clone();
	// cv::resize(input_, input_, cv::Size(), 0.3, 0.3);
	cv::cvtColor(input_, gray_, CV_BGR2GRAY);
	output_ = input_.clone();
	faces_.clear();
	genders_.clear();
	smile_flags_.clear();
}

void EnsembleDetector::AddPadding(cv::Rect& r, int padding){

	r.x -= padding;
	r.width += 2 * padding;
	r.y -= padding;
	r.height += 2 * padding;

	r.x = r.x>=0? r.x: 0;
	r.y = r.y>=0? r.y: 0;
}

void EnsembleDetector::AppendRectangles(
    vector<cv::Rect>& old_list,
    vector<cv::Rect>& new_list) {

    vector<cv::Rect> rects_to_be_appended;
    for(auto new_rect : new_list) {
        bool flag = false;
        for(auto old_rect : old_list) {
        	if (!allowed(old_rect, new_rect)) {
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

bool EnsembleDetector::allowed(cv::Rect& old_rect, cv::Rect& new_rect) {
	auto insec =  old_rect & new_rect;
	int min_area = min(old_rect.width * old_rect.height, new_rect.width * new_rect.height);
	float rate = 1.f * insec.width * insec.height / min_area;
	return (rate < 0.25f);
}

bool EnsembleDetector::Detect() {
	float rate = 1;
	cv::Mat img = gray_.clone();
	vector<cv::Rect> hrects;
	vector<cv::Rect> lrects;
	while(true) {
		cv::Size size = img.size();
		if (size.width < FACESIZE || size.height< FACESIZE)
			break;

		vector<cv::Rect> _hrects;
		vector<cv::Rect> _lrects;
		ScanImage(img, _hrects, _lrects);
		groupRectangles(_hrects, 1, EPS);

		for (cv::Rect &rect: _hrects) {
			rect.x = round(rect.x * rate);
			rect.y = round(rect.y * rate);
			rect.width = round(rect.width * rate);
			rect.height = round(rect.height * rate);
		}
		for (cv::Rect &rect: _lrects) {
			rect.x = round(rect.x * rate);
			rect.y = round(rect.y * rate);
			rect.width = round(rect.width * rate);
			rect.height = round(rect.height * rate);
		}

		AppendRectangles(hrects, _hrects);
		for (auto r: _lrects)
		{
			lrects.push_back(r);
		}
		rate *= SCALERATE;
		cv::resize(gray_, img, cv::Size(), 1 / rate, 1 / rate);
	}
	for (auto r: hrects)
	{
		faces_.push_back(r);
	}
	groupRectangles(lrects, 2, 0.51);
	for (int i = 0; i < lrects.size(); ++i) {
		cv::Rect& r1 = lrects[i];
		bool flag = false;
		for (int j = 0; j < hrects.size(); ++j)
		{
			cv::Rect& r2 = hrects[j];
			if (!allowed(r1, r2)) {
				flag = true;
				break;
			}
		}
		for (int j = 0; (j != i) && (j < lrects.size()); ++j) {
			cv::Rect& r2 = lrects[j];
			if (!allowed(r1, r2)) {
				flag = true;
				break;
			}
		}
		if (!flag) {
			faces_.push_back(r1);
		}
	}
	faces_ = RemoveTooLargeRectangles(faces_, 2);
	CalcProperties();
	return true;
}

void EnsembleDetector::CalcProperties() {
	cv::Rect whole;
	whole.x = 0;
	whole.y = 0;
	whole.width = gray_.size().width;
	whole.height = gray_.size().height;
	for (auto fr: faces_) {
		auto r = fr & whole;
		cv::Mat patch = gray_(r).clone();
		cv::resize(patch, patch, cv::Size(FACESIZE, FACESIZE));
		float gscore = predictor_gender_->Predict(patch)[1];
		float sscore = predictor_smile_->Predict(patch)[1];
		if (gscore > GENDER_THRESHOLD) {
			genders_.push_back(1);
		} else {
			genders_.push_back(0);
		}
		if (sscore > SMILE_THRESHOLD) {
			smile_flags_.push_back(1);
		} else {
			smile_flags_.push_back(0);
		}
	}
	for (int i = 0; i < faces_.size(); ++i) {
		auto r = faces_[i] & whole;
		if (genders_[i]) {
			cv::rectangle(output_, r, cv::Scalar(255, 0, 0), 2);
		} else {
			cv::rectangle(output_, r, cv::Scalar(0, 255, 0), 2);
		}
		if (smile_flags_[i]) {
			cv::Point center;
			center.x = r.x + (r.width + 1) / 2;
			center.y = r.y + (r.height + 1) / 2;
			cv::circle(output_, center, r.width / 2, cv::Scalar(0, 0, 255), 2);
			// cv::putText(output_, "SMILE", cv::Point(r.x, r.y + r.height), 
			// 	cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 2);
		}
	}
	// cv::Size in_size = input_.size();
	// in_size.width += 5;
	// in_size.height += 5;
	// cv::resize(output_, output_, in_size);
	// for (cv::Rect rect: faces_) {
	// 	cv::rectangle(output_, rect, cv::Scalar(0, 255, 0));
	// }
}

vector<Face> EnsembleDetector::GetFaces() {
	std::cout << "EnsembleDetector::GetFaces" << std::endl;
	vector<Face> results(faces_.size());
	for (int i = 0; i < faces_.size(); ++i)
	{
		results[i].rect = faces_[i];
		results[i].gender = genders_[i];
		results[i].is_smile = smile_flags_[i];
	}
	return results;
}

cv::Mat EnsembleDetector::GetOutput() {
	return output_;
}

void EnsembleDetector::ScanImage(cv::Mat &img, vector<cv::Rect> &highrects, vector<cv::Rect> &lowrects) {
	std::cout << "EnsembleDetector::ScanImage" << std::endl;
	cv::Size size = img.size();
	for (int i = 0; i < img.rows; i+= STRIDE) {
		// vector<float> row_score_map;
		for (int j = 0; j < img.cols; j += STRIDE) {
			cv::Point p(j, i);
			cv::Rect rect = _roi(p);
			if (!_check(rect, size)) {
				continue;
			}
			cv::Mat patch = img(rect);
			float score1 = predictor1_->Predict(patch)[1];
			float score2 = predictor2_->Predict(patch)[1];
			if (score1 < 0.5 && score2 < 0.5) {
				continue;
			}
			if ((score1 > SCORETHRESHOLD) || (score2 > SCORETHRESHOLD)
				|| ((score1 > 0.5) && (score2 > 0.5))) {
				highrects.push_back(rect);
			}
			else {
				lowrects.push_back(rect);
			}
		}
	}
}
