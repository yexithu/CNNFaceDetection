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
