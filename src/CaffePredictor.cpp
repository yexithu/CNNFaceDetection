#include "CaffePredictor.hpp"

using namespace caffe;
using std::string;

CaffePredictor::CaffePredictor(const string& model_file, const string& trained_file) {
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}

vector<float> CaffePredictor::Predict(const cv::Mat &img) {
    Blob<float>* input_layer = net_->input_blobs()[0];
    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels);

    net_->Forward();
    /* Copy the output layer to a std::vector */
    Blob<float>* output_layer = net_->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();
    return std::vector<float>(begin, end);
}


void CaffePredictor::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void CaffePredictor::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  // cv::Mat sample_resized;
  // if (sample.size() != input_geometry_)
  //   cv::resize(sample, sample_resized, input_geometry_);
  // else
  //   sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample.convertTo(sample_float, CV_32FC3);
  else
    sample.convertTo(sample_float, CV_32FC1);

  // cv::Mat sample_normalized;
  // cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_float, *input_channels);
  // CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
  //       == net_->input_blobs()[0]->cpu_data())
  //   << "Input channels are not wrapping the input layer of the network.";
}
