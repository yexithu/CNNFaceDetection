#include <opencv2/opencv.hpp>
#include <vector>
using std::vector;

inline bool operator <= (const cv::Rect& r1, const cv::Rect& r2) {
    // rect1 subseteq rect2
    return (r1 & r2) == r1;
}

void AppendRectangles(
    vector<cv::Rect>& old_list,
    vector<cv::Rect>& new_list)
{
    vector<cv::Rect> rects_to_be_appended;
    for(auto new_rect : new_list){
        bool overlap = false;
        for(auto old_rect : old_list){
            auto union_rect = new_rect | old_rect;
            if(union_rect == new_rect){
                // new covers old
                overlap = true;
                break;
            }
        }
        if(!overlap){
            rects_to_be_appended.push_back(new_rect);
        }
    }

    for(auto i : rects_to_be_appended){
        old_list.push_back(i);
    }
    return;
}


using std::cout;
using std::endl;

vector<cv::Rect> _RemoveTooLargeRectangles(vector<cv::Rect>& rects, double k = 3){
    vector<double> keys;

    for(auto& rect : rects){
        keys.push_back((rect).width);
    }
    // cout << "n: " << rects.size();
    double sum = std::accumulate(keys.begin(), keys.end(), 0.0);
    double mean = sum / keys.size();
    double sq_sum = std::inner_product(keys.begin(), keys.end(), keys.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / keys.size() - mean * mean);
    double upper_bound = mean + k * stdev;

    // cout << "sum, mu, signa: " << sum << ',' << mean << ',' << stdev << endl;

    vector<cv::Rect> result;
    for(auto i = 0u; i < rects.size(); i ++){
        // cout << keys[i] << ',';
        if(keys[i] < upper_bound){
            result.push_back(rects[i]);
        }
    }
    return result;
}

vector<cv::Rect> RemoveTooLargeRectangles(vector<cv::Rect>& rects, double k = 3){
    while(1){
        auto result = _RemoveTooLargeRectangles(rects, k);
        if(result.size() == rects.size()){
            return result;
        }
        rects = result;
    }
}
