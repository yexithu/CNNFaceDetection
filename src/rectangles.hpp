#include <opencv2/opencv.hpp>
#include <vector>
using std::vector;


vector<cv::Rect> AppendRectangles(
    vector<cv::Rect>& old_list,
    vector<cv::Rect>& new_list)
{
    vector<cv::Rect> result = old_list;
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
            result.push_back(new_rect);
        }
    }
    return result;
}
