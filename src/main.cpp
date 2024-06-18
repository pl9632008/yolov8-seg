#include "rail.h"
#include <opencv2/opencv.hpp>
#include <iostream>
 
int main() {

    infer_rail infer;

    std::string config_path = "/home/nvidia/wjd/NG_LRIALS_v1.0/infer/conf/Config.ini";
    auto configurations = infer.readConfig(config_path);

    RailInfo rail_info;
    MarkInfo mark_info_near;
    MarkInfo mark_info_far;

    MarkInfo back_mark_info_near;
    MarkInfo back_mark_info_far;


    for(auto config : configurations){
        

        if(config.section == "[ConfAI]"){
            
            rail_info.width = std::stod(config.data["width"]);
            rail_info.height = std::stod(config.data["height"]);
            rail_info.y_limit = std::stod(config.data["y_limit"]);
            rail_info.conf_threshold = std::stod(config.data["conf_threshold"]);
            rail_info.nms_threshold = std::stod(config.data["nms_threshold"]);
            rail_info.mask_threshold = std::stod(config.data["mask_threshold"]);

            rail_info.matrix_config_path = config.data["matrix_config_path"];
            rail_info.yolo_engine_path = config.data["yolo_engine_path"];
          
            mark_info_near.p_left_top_x = std::stod(config.data["p_left_top_x"]);
            mark_info_near.p_left_top_y = std::stod(config.data["p_left_top_y"]);
            mark_info_near.p_left_bottom_x = std::stod(config.data["p_left_bottom_x"]); 
            mark_info_near.p_left_bottom_y = std::stod(config.data["p_left_bottom_y"]);
            mark_info_near.p_right_top_x = std::stod(config.data["p_right_top_x"]);
            mark_info_near.p_right_top_y = std::stod(config.data["p_right_top_y"]);
            mark_info_near.p_right_bottom_x = std::stod(config.data["p_right_bottom_x"]);
            mark_info_near.p_right_bottom_y = std::stod(config.data["p_right_bottom_y"]);
            mark_info_near.offset_bottom = std::stod(config.data["offset_bottom"]);
            mark_info_near.offset_top = std::stod(config.data["offset_top"]);
            mark_info_near.bottom_diff = std::stod(config.data["bottom_diff"]);
            mark_info_near.top_diff = std::stod(config.data["top_diff"]);


            mark_info_far.p_left_top_x = std::stod(config.data["p_left_top_far_x"]);
            mark_info_far.p_left_top_y = std::stod(config.data["p_left_top_far_y"]);
            mark_info_far.p_left_bottom_x = std::stod(config.data["p_left_bottom_far_x"]); 
            mark_info_far.p_left_bottom_y = std::stod(config.data["p_left_bottom_far_y"]);
            mark_info_far.p_right_top_x = std::stod(config.data["p_right_top_far_x"]);
            mark_info_far.p_right_top_y = std::stod(config.data["p_right_top_far_y"]);
            mark_info_far.p_right_bottom_x = std::stod(config.data["p_right_bottom_far_x"]);
            mark_info_far.p_right_bottom_y = std::stod(config.data["p_right_bottom_far_y"]);
            mark_info_far.offset_bottom = std::stod(config.data["offset_bottom_far"]);
            mark_info_far.offset_top = std::stod(config.data["offset_top_far"]);
            mark_info_far.bottom_diff = std::stod(config.data["bottom_diff_far"]);
            mark_info_far.top_diff = std::stod(config.data["top_diff_far"]);

        }
   
    }

    infer.init(rail_info, mark_info_near);
    //infer.init(rail_info, mark_info_far);

    cv::VideoCapture cap_near("../jin.avi"); // 替换为你的视频文件路径
    
    int width = cap_near.get(cv::CAP_PROP_FRAME_WIDTH);             //帧宽度
    int height = cap_near.get(cv::CAP_PROP_FRAME_HEIGHT);           //帧高度
    int totalFrames = cap_near.get(cv::CAP_PROP_FRAME_COUNT);       //总帧数
    int frameRate = cap_near.get(cv::CAP_PROP_FPS);                 //帧率 x frames/s
    int fcc = cap_near.get(cv::CAP_PROP_FOURCC);

    std::cout << "视频宽度： " << width << std::endl;
    std::cout << "视频高度： " << height << std::endl;
    std::cout << "视频总帧数： " << totalFrames << std::endl;
    std::cout << "帧率： " << frameRate << std::endl;

    // cv::VideoWriter wri;
    // wri.open("../result.avi", fcc, frameRate, cv::Size(width, height));

    while(1) {
     
        cv::Mat frame_near;
        cap_near>>frame_near;
 
        if(frame_near.empty()){
            break;
        }


        std::vector<point_set> po = infer.infer_out(frame_near);
        
        cv::Mat img1_rail_mk = infer.img1_rail_mask_;

        cv::Scalar color1 = cv::Scalar(56,0,255);
        cv::Scalar color2 = cv::Scalar(0,255,56);

        for(int row = 0 ; row < frame_near.rows ; row++){

            for(int col = 0 ; col < frame_near.cols ; col++){

                if( !img1_rail_mk.empty() && img1_rail_mk.at<uint8_t>(row,col) == 255){
                    frame_near.at<cv::Vec3b>(row,col)[0] = cv::saturate_cast<uint8_t>( 0.5 * frame_near.at<cv::Vec3b>(row,col)[0] + 0.5 * color1[0] );
                    frame_near.at<cv::Vec3b>(row,col)[1] = cv::saturate_cast<uint8_t>( 0.5 * frame_near.at<cv::Vec3b>(row,col)[1] + 0.5 * color1[1] );
                    frame_near.at<cv::Vec3b>(row,col)[2] = cv::saturate_cast<uint8_t>( 0.5 * frame_near.at<cv::Vec3b>(row,col)[2] + 0.5 * color1[2] );
                }

            }

        }
        

        int offset_bottom ;
        int offset_top;
        cv::Point left_top;
        cv::Point left_bottom;
        cv::Point right_top;
        cv::Point right_bottom;


        int offset_bottom_far;
        int offset_top_far;
        cv::Point left_top_far;
        cv::Point left_bottom_far;
        cv::Point right_top_far;
        cv::Point right_bottom_far;



        offset_bottom = infer.offset_bottom_;
        offset_top = infer.offset_top_;
        left_top = infer.p_left_top_;
        left_bottom = infer.p_left_bottom_;
        right_top = infer.p_right_top_;
        right_bottom = infer.p_right_bottom_;

        offset_bottom_far = infer.offset_bottom_far_;
        offset_top_far = infer.offset_top_far_;
        left_top_far = infer.p_left_top_far_;
        left_bottom_far = infer.p_left_bottom_far_;
        right_top_far = infer.p_right_top_far_;
        right_bottom_far = infer.p_right_bottom_far_;



        if(1){

            cv::line(frame_near, cv::Point(left_top.x - offset_top, left_top.y), cv::Point(left_bottom.x - offset_bottom, left_bottom.y), cv::Scalar(255,0,0),3);
            cv::line(frame_near, cv::Point(left_top.x             , left_top.y), cv::Point(left_bottom.x                , left_bottom.y), cv::Scalar(0,0,255),3);
            cv::line(frame_near, cv::Point(left_top.x + offset_top, left_top.y), cv::Point(left_bottom.x + offset_bottom, left_bottom.y), cv::Scalar(255,0,0),3);
            cv::line(frame_near, cv::Point(right_top.x - offset_top, right_top.y), cv::Point(right_bottom.x - offset_bottom, right_bottom.y), cv::Scalar(255,0,0),3);
            cv::line(frame_near, cv::Point(right_top.x             , right_top.y), cv::Point(right_bottom.x                , right_bottom.y), cv::Scalar(0,0,255),3);
            cv::line(frame_near, cv::Point(right_top.x + offset_top, right_top.y), cv::Point(right_bottom.x + offset_bottom, right_bottom.y), cv::Scalar(255,0,0),3);
        }else{

            cv::line(frame_near, cv::Point(left_top_far.x - offset_top_far, left_top_far.y), cv::Point(left_bottom_far.x - offset_bottom_far, left_bottom_far.y), cv::Scalar(255,0,0),3);
            cv::line(frame_near, cv::Point(left_top_far.x                 , left_top_far.y), cv::Point(left_bottom_far.x                    , left_bottom_far.y), cv::Scalar(0,0,255),3);
            cv::line(frame_near, cv::Point(left_top_far.x + offset_top_far, left_top_far.y), cv::Point(left_bottom_far.x + offset_bottom_far, left_bottom_far.y), cv::Scalar(255,0,0),3);
            cv::line(frame_near, cv::Point(right_top_far.x - offset_top_far, right_top_far.y), cv::Point(right_bottom_far.x - offset_bottom_far, right_bottom_far.y), cv::Scalar(255,0,0),3);
            cv::line(frame_near, cv::Point(right_top_far.x                 , right_top_far.y), cv::Point(right_bottom_far.x                    , right_bottom_far.y), cv::Scalar(0,0,255),3);
            cv::line(frame_near, cv::Point(right_top_far.x + offset_top_far, right_top_far.y), cv::Point(right_bottom_far.x + offset_bottom_far, right_bottom_far.y), cv::Scalar(255,0,0),3);
        }
    

        

        for(auto i : po){

            if(i.type == 1 ){
                cv::rectangle(frame_near,cv::Point(i.xmin,i.ymin),cv::Point(i.xmax,i.ymax),cv::Scalar(255,0,0),3);
            }
             if(i.type == 2 ){
                cv::rectangle(frame_near,cv::Point(i.xmin,i.ymin),cv::Point(i.xmax,i.ymax),cv::Scalar(0,0,255),3);
            }

        }
    

        cv::line(frame_near, cv::Point(0 , infer.lidar_distance_near_), cv::Point(1919, infer.lidar_distance_near_), cv::Scalar(0,0,0), 4);

        cv::imwrite("../frame_near.jpg",frame_near);

        // wri<<frame_near;

    }
 
 
    // wri.release();
    // cap_near.release();
    // cv::destroyAllWindows();

}
