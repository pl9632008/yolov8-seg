
#include <chrono>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <atomic>
#include <fstream>
#include <cmath>
#include <NvInfer.h>
#include <algorithm>
#include <nlohmann/json.hpp>
#include "ini.h"
using namespace nvinfer1;

using json = nlohmann::json;


struct Object{

    int class_label;
    std::string label;
    cv::Rect rect;
    float prob;
    cv::Mat mask;

};

struct point_set
{
    int xmin;      // left
    int xmax;      // right
    int ymin;      // top
    int ymax;      // bottom
    int label;     // 标签类别 0:轨道 1:人 2:车 3:电动车自行车
    int type;      // 预警类别 0:无需处理 1:绘制目标框 2:语音告警
    int proximity; // 用于远近摄像头判断
    float confidence;

};


struct RailInfo{
    int height;
    int width;
    int y_limit;
    float conf_threshold;
    float mask_threshold;
    float nms_threshold;
    std::string matrix_config_path;
    std::string yolo_engine_path;


};

struct MarkInfo{
 
    int p_left_top_x;
    int p_left_top_y;

    int p_left_bottom_x;
    int p_left_bottom_y;

    int p_right_top_x;
    int p_right_top_y;

    int p_right_bottom_x;
    int p_right_bottom_y;

    int offset_bottom;
    int offset_top;

    int bottom_diff;
    int top_diff;

};


struct Configuration {
    std::string section;
    std::map<std::string, std::string> data;
};


class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} ;



class infer_rail: public ini::iniReader

{
public:
    infer_rail();
    ~infer_rail();
    void loadEngine(const std::string& path );
    cv::Mat preprocessImg(cv::Mat& img, const int& input_w, const int& input_h, int& padw, int& padh) ;
    std::vector<Object> doInference(cv::Mat & img);
    void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects);
    cv::Mat findLargestContour(cv::Mat & mask);
    int nearRailNew(point_set & target, cv::Mat & rail_mask);
    float point_distance(float left, float top, float right, float bottom);
    int findEdge(cv::Mat & rail_mask, const int & far_flag);
    std::vector<std::vector<cv::Point>> splitRailPoints(std::vector<cv::Point> & pts, const int & len);
    std::vector<bool> findAbnormalCluster( std::vector<std::vector<cv::Point>> & splits, const int & flag, const int & far_flag);
    int findMaxPosition(std::vector<bool> & temp);
    void setRailInfo(RailInfo & rail_info);
    void setMarkInfoNear(MarkInfo & mark_info_near);
    void fitLines(const int & flag);
    std::vector<Configuration> readConfig(std::string & configPath);
    void init(RailInfo & rail_info, MarkInfo & mark_info_near);
    std::vector<point_set> infer_out(cv::Mat & image1);

    void res2labelme(cv::Mat & img, std::vector<Object> & objs);
    std::string video_name_;
    std::string out_path_;


    Logger logger_;
    std::unique_ptr<IRuntime> runtime_;
    std::unique_ptr<ICudaEngine> engine_;
    std::unique_ptr<IExecutionContext> context_;

    std::vector<std::string> class_names_{"rail", "person", "car", "vehicle"};
    int height_ = 720;
    int width_ = 1280;
    int y_limit_ = 680;
    const char * images_ = "images";
    const char * output0_ = "output0";  // detect
    const char * output1_ = "output1"; // mask
    int BATCH_SIZE = 1;
    int CHANNELS = 3; 
    int INPUT_H = 640;
    int INPUT_W = 640;
    int CLASSES = 4;
    int NUM_BOXES = 8400;
    int CHANNELS_OUT = 32;
    int MASK_WIDTH = 160;
    int MASK_HEIGHT = 160;
    float CONF_THRESHOLD = 0.5;
    float NMS_THRESHOLD = 0.7;
    float MASK_THRESHOLD = 0.5;
    float * images_arr_;
    float * output0_arr_;
    float * output1_arr_;
    int lidar_distance_near_ ;

    cv::Vec4f cal_line1_;
    cv::Vec4f cal_line2_;
    cv::Vec4f cal_line3_;
    cv::Vec4f cal_line4_;
    cv::Vec4f cal_line1_far_;
    cv::Vec4f cal_line2_far_;
    cv::Vec4f cal_line3_far_;
    cv::Vec4f cal_line4_far_;


 //近
    cv::Point p_left_top_ = {cv::Point(1082,362)};
    cv::Point p_left_bottom_ = {cv::Point(804,1068)};
    cv::Point p_right_top_ = {cv::Point(1114,370)};
    cv::Point p_right_bottom_= {cv::Point(1467,1050)};
    int offset_bottom_ = 60;
    int offset_top_ = 10;
    int bottom_diff_ = 662;
    int top_diff_ = 32;
    cv::Point p1_;
    cv::Point p2_;
    cv::Vec4f line1_;
    cv::Vec4f line2_;
    cv::Vec4f line3_;
    cv::Vec4f line4_;

//远
    cv::Point p_left_top_far_ = {cv::Point(1268,34)};
    cv::Point p_left_bottom_far_ = {cv::Point(826,1064)};
    cv::Point p_right_top_far_ = {cv::Point(1312,34)};
    cv::Point p_right_bottom_far_= {cv::Point(1875,1064)};
    int offset_bottom_far_ = 90;
    int offset_top_far_ = 20;
    int bottom_diff_far_ = 1049;
    int top_diff_far_ = 44;
    cv::Point p1_far_;
    cv::Point p2_far_;
    cv::Point p1_origin_far_;
    cv::Point p2_origin_far_;
    cv::Mat img_rail_mask_far_;
    cv::Vec4f line1_far_;
    cv::Vec4f line2_far_;
    cv::Vec4f line3_far_;
    cv::Vec4f line4_far_;



 //Back近
    cv::Point back_p_left_top_ = {cv::Point(1082,362)};
    cv::Point back_p_left_bottom_ = {cv::Point(804,1068)};
    cv::Point back_p_right_top_ = {cv::Point(1114,370)};
    cv::Point back_p_right_bottom_= {cv::Point(1467,1050)};
    int back_offset_bottom_ = 60;
    int back_offset_top_ = 10;
    int back_bottom_diff_ = 662;
    int back_top_diff_ = 32;
    cv::Vec4f back_line1_;
    cv::Vec4f back_line2_;
    cv::Vec4f back_line3_;
    cv::Vec4f back_line4_;

//Back远
    cv::Point back_p_left_top_far_ = {cv::Point(1268,34)};
    cv::Point back_p_left_bottom_far_ = {cv::Point(826,1064)};
    cv::Point back_p_right_top_far_ = {cv::Point(1312,34)};
    cv::Point back_p_right_bottom_far_= {cv::Point(1875,1064)};
    int back_offset_bottom_far_ =90;
    int back_offset_top_far_ = 20;
    int back_bottom_diff_far_ = 1049;
    int back_top_diff_far_ = 44;
    cv::Vec4f back_line1_far_;
    cv::Vec4f back_line2_far_;
    cv::Vec4f back_line3_far_;
    cv::Vec4f back_line4_far_;


//pds1.avi
    // cv::Point p_left_top_ = {cv::Point(638,127)};
    // cv::Point p_left_bottom_ = {cv::Point(475,714)};
    // cv::Point p_right_top_ = {cv::Point(674,127)};
    // cv::Point p_right_bottom_= {cv::Point(1064,714)};
    // int offset_bottom_ = 60;
    // int offset_top_ = 10;
    // int bottom_diff_ = 589;
    // int top_diff_ = 36;

//外部可视化
    cv::Mat img1_rail_mask_;
    cv::Mat img2_rail_mask_;

};