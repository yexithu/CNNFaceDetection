#ifndef DETECTOR_HPP
#define DETECTOR_HPP
#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
class CaffePredictor;

class Detector {
public:
	Detector(const std::string& model_file, const std::string& trained_file);
	void Input(const cv::Mat img);
	bool Detect();
	std::vector<cv::Rect> GetFaces();
	std::vector<cv::Rect> ScanImage(cv::Mat &img);
	cv::Mat GetOutput();

	inline bool overlap(cv::Rect& x, cv::Rect& y){
		auto i = x & y;
		return i.width != 0 || i.height != 0;
	}

	inline bool cover(cv::Rect& x, cv::Rect& y){
		//x covers y, y subseteq x
		return (x & y) == y;
	}

private:
	cv::Mat input_;
	cv::Mat gray_;
	cv::Mat output_;
	std::vector<cv::Rect> faces_;

	caffe::shared_ptr<CaffePredictor > predictor_;

	const int FACESIZE;
	const int HALFSIZE;
	const int STRIDE;
	const float SCALERATE;
	const int GROUPTHRESHOLD;

	void AddPadding(cv::Rect&, int);
	void AppendRectangles(std::vector<cv::Rect>& old_list,  std::vector<cv::Rect>& new_list);

	inline cv::Rect _roi(cv::Point& center) {
		return cv::Rect(center.x - HALFSIZE, center.y - HALFSIZE, FACESIZE, FACESIZE);
	}

	inline bool _check(cv::Rect& rect, cv::Size &size) {
		return ((rect.x >= 0) && (rect.y >= 0) &&
			    ((rect.x + rect.width) <= size.width) &&
				((rect.y + rect.height) <= size.height));
	}

};
#endif
