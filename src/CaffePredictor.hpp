#ifndef CAFFE_PREDICTOR_HPP
#define CAFFE_PREDICTOR_HPP
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>



class CaffePredictor {
public:
	CaffePredictor(const std::string& model_file, const std::string& trained_file);
	std::vector<float> Predict(const cv::Mat &img);
private:
	void WrapInputLayer(std::vector<cv::Mat>* input_channels);
	void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);
	caffe::shared_ptr<caffe::Net<float> > net_;
	cv::Size input_geometry_;
  	int num_channels_;
};
#endif