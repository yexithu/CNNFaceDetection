#include "Detector.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <fstream>

using namespace std;
const int FACESIZE = 25;
const float SCALERATE = 1.5;


vector<cv::Rect> Detect(cv::Mat& gray_, Detector& detector) {
	float rate = 1;	
	cv::Mat img = gray_.clone();
	auto faces_ = vector<cv::Rect>();

	while(true) {
		cv::Size size = img.size();
		if (size.width < FACESIZE || size.height< FACESIZE)
			break;

		vector<cv::Rect> rects = detector.ScanImage(img);
		for (cv::Rect &rect: rects) {
			rect.x = round(rect.x * rate);
			rect.y = round(rect.y * rate);
			rect.width = round(rect.width * rate);
			rect.height = round(rect.height * rate);
		}
		
		for (auto r: rects) 
			faces_.push_back(r);
		rate *= SCALERATE;
		cv::resize(gray_, img, cv::Size(), 1 / rate, 1 / rate);
	}
	return faces_;

}

int area(cv::Rect& _){
	return _.width * _.height;
}

bool negative_sample(cv::Rect& detected, cv::Rect& label){
	auto _cover = Detector::cover(label, detected);
	auto _no_overlap = ! Detector::overlap(detected, label);
	auto _area_small = area(detected) < (area(label) / 4);
	return _no_overlap || (_cover && _area_small);
}

string remove_extension(string& filename){
	return filename.substr(0, filename.length()-4);
}

void mkdir(string _){
	system(("mkdir " + _).c_str());
}


int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
            << " deploy.prototxt network.caffemodel"
            << " data_tsv data_root output_root" << std::endl;
        return 1;
    }
    
    ::google::InitGoogleLogging(argv[0]);

    string model_file   = argv[1];
    string trained_file = argv[2];
    string data_tsv    = argv[3];
    string data_root  = argv[4];
    string output_root = argv[5];

    //prepare

    system(("rm -r " + output_root).c_str());
    mkdir(output_root);
    mkdir(output_root + "/0");
    mkdir(output_root + "/3");
    mkdir(output_root + "/2");




    Detector detector(model_file, trained_file);
    
    ifstream infile;
    infile.open(data_tsv.c_str(), ios::in);
    string line;
    char buf[30];
    int face_id;
    cv::Rect buf_rect;

    vector<string> filenames;
    vector<cv::Rect> rectangles;
    while(getline(infile, line)) {
        sscanf(line.c_str(), "%s\t%d\t%d\t%d\t%d\t%d", buf, 
        	&face_id, &buf_rect.x, &buf_rect.y, &buf_rect.width, &buf_rect.height);
        // cout << buf << " " << buf_rect.x << " " << buf_rect.y << " " << buf_rect.width << " " << buf_rect.height << endl;
        filenames.push_back(buf);
        rectangles.push_back(buf_rect);
    }
    infile.close();
    auto n = filenames.size();
    for(auto i=0; i < n ; i++){
    	
    	auto filename = filenames[i];
    	cout << data_root + filename << endl;
    	auto gray_ = cv::imread(data_root + filename, 0);

    	auto detected = Detect(gray_, detector);
    	vector<cv::Rect> negatives;

    	for(auto& detected_rect : detected){
    		if(!negative_sample(detected_rect, rectangles[i])){
    			continue;
    		}
    		negatives.push_back(detected_rect);
    	}

    	string output_filename_prefix = output_root + remove_extension(filename) + "/";
    	mkdir(output_filename_prefix);
    	for(auto i=0; i<negatives.size(); i++){
    		//crop
    		string output_filename = output_filename_prefix + to_string(i) + ".jpg";
    		cout << output_filename << endl;
    		auto croped = gray_(negatives[i]);
    		cv::imwrite(output_filename, croped);
    	}
    }
    return 0;
}