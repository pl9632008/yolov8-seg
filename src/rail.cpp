#include "rail.h"

infer_rail::infer_rail(){

    images_arr_ = new float[BATCH_SIZE * CHANNELS * INPUT_H * INPUT_W];
    output0_arr_ = new float[BATCH_SIZE * NUM_BOXES * (4 + CLASSES + CHANNELS_OUT) ];
    output1_arr_ = new float[BATCH_SIZE * CHANNELS_OUT * MASK_HEIGHT * MASK_WIDTH ];

}
infer_rail::~infer_rail(){
    delete[] images_arr_;
    delete[] output0_arr_;
    delete[] output1_arr_;
}


void infer_rail::loadEngine(const std::string& path ) {
    size_t size{ 0 };
    char* trtModelStream{ nullptr };
    std::ifstream file(path, std::ios::binary);

    if (file.good()) {
        file.seekg(0, std::ios::end);
        size = file.tellg();
        file.seekg(0, std::ios::beg);
        trtModelStream = new char[size];
        file.read(trtModelStream, size);
        file.close();
    }

    std::unique_ptr<IRuntime> runtime(createInferRuntime(logger_));
    std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(trtModelStream,size));
    std::unique_ptr<IExecutionContext>context(engine->createExecutionContext());

    runtime_ = std::move(runtime);
    engine_  = std::move(engine);
    context_ = std::move(context);

    delete[] trtModelStream;
}


cv::Mat infer_rail::preprocessImg(cv::Mat& img, const int& input_w, const int& input_h, int& padw, int& padh) {
    int w, h, x, y;
    float r_w = input_w / (img.cols * 1.0);
    float r_h = input_h / (img.rows * 1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    }
    else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(127, 127, 127));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

    padw = (input_w - w) / 2;
    padh = (input_h - h) / 2;
    return out;
}


cv::Mat infer_rail::findLargestContour(cv::Mat & mask)
{
	cv::Mat result(mask.size(), CV_8UC1, cv::Scalar(0));

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	double maxArea = 0;
	int maxAreaIdx = -1;
	for (int i = 0; i < contours.size(); ++i)
	{
		double area = cv::contourArea(contours[i]);
		if (area > maxArea)
		{
			maxArea = area;
			maxAreaIdx = i;
		}
	}


	if (maxAreaIdx >= 0)
	{
		cv::drawContours(result, contours, maxAreaIdx, cv::Scalar(255), cv::FILLED);
	}
	return result;
}

std::vector<Object> infer_rail::doInference(cv::Mat & img){

    int input_index = engine_->getBindingIndex(images_);
    int output0_index = engine_->getBindingIndex(output0_);
    int output1_index = engine_->getBindingIndex(output1_);
    
    void* buffers[3];
    cudaMalloc(&buffers[input_index],   BATCH_SIZE * CHANNELS * INPUT_H * INPUT_W * sizeof(float));
    cudaMalloc(&buffers[output0_index], BATCH_SIZE * NUM_BOXES * (4 + CLASSES + CHANNELS_OUT) * sizeof(float));
    cudaMalloc(&buffers[output1_index], BATCH_SIZE * CHANNELS_OUT * MASK_HEIGHT * MASK_WIDTH * sizeof(float));

    int padw = 0;
    int padh = 0;
    cv::Mat pr_img = preprocessImg(img, INPUT_W, INPUT_H, padw, padh);  


    for (int i = 0; i < INPUT_W * INPUT_H; i++) {
        images_arr_[i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
        images_arr_[i + INPUT_W * INPUT_H] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
        images_arr_[i + 2 * INPUT_W * INPUT_H] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(buffers[input_index], images_arr_, BATCH_SIZE * CHANNELS * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream);
    context_->enqueueV2(buffers, stream, nullptr);
    cudaMemcpyAsync(output0_arr_, buffers[output0_index], BATCH_SIZE * NUM_BOXES * (4 + CLASSES + CHANNELS_OUT) * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(output1_arr_, buffers[output1_index], BATCH_SIZE * CHANNELS_OUT * MASK_HEIGHT * MASK_WIDTH * sizeof(float), cudaMemcpyDeviceToHost, stream);


    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(buffers[input_index]);
    cudaFree(buffers[output0_index]);
    cudaFree(buffers[output1_index]);

    float ratio_w = INPUT_W / (img.cols * 1.0);
    float ratio_h = INPUT_H / (img.rows * 1.0);

    int net_width = 4 + CLASSES + CHANNELS_OUT;

    std::vector<int> classIds;						  
    std::vector<float> scores;					 
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<float>> temp_result;

    float *pdata = output0_arr_;

    for (int i = 0; i < NUM_BOXES; i++) {
        float* score_ptr = std::max_element(pdata + 4, pdata + 4 + CLASSES);
        float box_score = *score_ptr;
        int label_index = score_ptr - (pdata + 4);

        if (box_score >= CONF_THRESHOLD) {

            float x = pdata[0]; 
            float y = pdata[1]; 
            float w = pdata[2];		   
            float h = pdata[3];	
            int l,r,t,b;

             if(ratio_h>ratio_w){
                    l = x-w/2.0;
                    r = x+w/2.0;
                    t = y-h/2.0-(INPUT_H-ratio_w*img.rows)/2;
                    b = y+h/2.0-(INPUT_H-ratio_w*img.rows)/2;
                    l=l/ratio_w;
                    r=r/ratio_w;
                    t=t/ratio_w;
                    b=b/ratio_w;
                }else{
                    l = x-w/2.0-(INPUT_W-ratio_h*img.cols)/2;
                    r = x+w/2.0-(INPUT_W-ratio_h*img.cols)/2;
                    t = y-h/2.0;
                    b = y+h/2.0;
                    l=l/ratio_h;
                    r=r/ratio_h;
                    t=t/ratio_h;
                    b=b/ratio_h;
                }	 

            classIds.push_back(label_index);
            scores.push_back(box_score);
            boxes.push_back(cv::Rect(l, t, r-l, b-t));

	    std::vector<float> temp(pdata + 4 + CLASSES, pdata + net_width);
            temp_result.push_back(temp);

        }
        pdata += net_width; 
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD, indices);
    
    std::vector<Object> objs;

    std::vector<std::vector<float>> temp_ans;
    for(auto idx : indices){
        int label_index= classIds[idx];
        std::string label = class_names_[label_index];
        cv::Rect rec = boxes[idx];
        float score = scores[idx];
        Object obj;
        obj.rect = rec;
        obj.prob = score;
        obj.label = label;
        obj.class_label = label_index;
        objs.push_back(obj);
        temp_ans.push_back(temp_result[idx]);

    }

   
    float r_w = 1.0 * INPUT_W / MASK_WIDTH;
    float r_h = 1.0 * INPUT_H / MASK_HEIGHT;
    cv::Rect roi( int( padw/r_w) ,int( padh/r_h),  int((INPUT_W - padw*2)/r_w), int((INPUT_W - padh*2)/r_h));

    for(int i = 0; i < indices.size(); i++){
        Object & obj = objs[i];
        cv::Mat mask = cv::Mat::zeros(MASK_HEIGHT, MASK_WIDTH, CV_32FC1);
        for(int p = 0; p < CHANNELS_OUT; p++){
            std::vector<float> temp(output1_arr_ + MASK_HEIGHT * MASK_WIDTH * p, output1_arr_ + MASK_HEIGHT * MASK_WIDTH *(p+1));
            float coeff = temp_ans[i][p];
            float *mp = (float *) mask.data;
            for(int j = 0; j < MASK_HEIGHT * MASK_WIDTH; j++){
                mp[j] += temp.data()[j]*coeff;
            }
        }

        //原始图到特征图的缩放比例,padding也要进行缩放
        cv::Mat dest;
        cv::exp(-mask,dest);
        dest = 1./(1.+ dest);
        dest= dest(roi);

        cv::Mat mask2 = cv::Mat::zeros(img.size(), CV_32FC1);
        resize(dest, mask2, img.size());
        obj.mask= cv::Mat::zeros(img.rows, img.cols, CV_8UC1);

        for(int y = 0; y < img.rows; y++){
            if(y<obj.rect.y || y > obj.rect.y + obj.rect.height){
                continue;
            }
            float* mp2 = mask2.ptr<float>(y);
            uchar* bmp = obj.mask.ptr<uchar>(y);
            for(int x = 0 ; x< img.cols; x++){
                if(x < obj.rect.x || x>obj.rect.x + obj.rect.width){
                    continue;
                }
                bmp[x] = mp2[x] > MASK_THRESHOLD ? 255 : 0;                
            }
        }
        obj.mask = findLargestContour(obj.mask);
    }

    //draw_objects(img, objs);
    return objs;

}

std::vector<point_set> infer_rail::infer_out(cv::Mat & image1){

    std::vector<point_set> box_set;
    cv::Mat current_near_rail_mask;
    bool far_has_rail = false;
    int lidar_near = height_ - 1;

    float min_dis = width_;
    int rail_correct = -1; // 正确的轨道序号

    std::vector<Object> objs = doInference(image1);
    std::vector<point_set> target_set; // 非轨道

    for (int i = 0; i < objs.size(); ++i){
      Object & obj = objs[i];
      if (obj.class_label == 0) {

          float dis = point_distance(obj.rect.x, obj.rect.y, obj.rect.x + obj.rect.width, obj.rect.y + obj.rect.height);
          if (dis > 0 && dis < min_dis){
            min_dis = dis;
            rail_correct = i;
        }
  
      }else if (obj.class_label == 1 || obj.class_label == 2 || obj.class_label == 3) // 人 汽车 电动车
      {
        target_set.push_back({int(obj.rect.x), int(obj.rect.x + obj.rect.width), int(obj.rect.y), int(obj.rect.y + obj.rect.height), obj.class_label, 0, 0, obj.prob}); // 记录位置，用于第二次循环判断人车是否在轨道上                                                                                              
      }
    }



    if(rail_correct != -1){

        current_near_rail_mask = objs[rail_correct].mask;
        img1_rail_mask_ = current_near_rail_mask;
        lidar_near = findEdge(current_near_rail_mask, 0);

        for (int tk = 0; tk < target_set.size(); tk++){
            point_set target = target_set[tk];
            cv::Mat box_mask = objs[rail_correct].mask;

            int type = nearRailNew(target, box_mask);
            target.type = type;
            if (target.type != 0){
                box_set.push_back(target);
            }
        }
    }


    lidar_distance_near_ = lidar_near;
    return box_set;
}



void infer_rail::init(RailInfo & rail_info, MarkInfo & mark_info_near){

    setRailInfo(rail_info);
    setMarkInfoNear(mark_info_near);
    loadEngine(rail_info.yolo_engine_path);
    fitLines(0);
    cal_line1_ = line1_;
    cal_line2_ = line2_;
    cal_line3_ = line3_;
    cal_line4_ = line4_;
}


void infer_rail::setRailInfo(RailInfo & rail_info){

  width_               =  rail_info.width;
  height_              =  rail_info.height;
  y_limit_             =  rail_info.y_limit;
  CONF_THRESHOLD       =  rail_info.conf_threshold;
  NMS_THRESHOLD        =  rail_info.nms_threshold;
  MASK_THRESHOLD       =  rail_info.mask_threshold;

}

void infer_rail::setMarkInfoNear(MarkInfo & mark_info_near){
    
    p_left_top_    =  cv::Point(mark_info_near.p_left_top_x    ,  mark_info_near.p_left_top_y    );
    p_left_bottom_ =  cv::Point(mark_info_near.p_left_bottom_x ,  mark_info_near.p_left_bottom_y );
  
    p_right_top_   =  cv::Point(mark_info_near.p_right_top_x    , mark_info_near.p_right_top_y   );
    p_right_bottom_=  cv::Point(mark_info_near.p_right_bottom_x , mark_info_near.p_right_bottom_y);
    
    offset_bottom_ =  mark_info_near.offset_bottom;
    offset_top_    =  mark_info_near.offset_top;
    bottom_diff_   =  mark_info_near.bottom_diff;
    top_diff_      =  mark_info_near.top_diff;

}



float infer_rail::point_distance(float left, float top, float right, float bottom)
{
  if (top < y_limit_ && bottom > y_limit_)
  {
    //float dis = fabs(((left + right) / 2) - 640);
    
    float dis = fabs(((left + right) / 2) - width_ /2.0 );
    
    return dis;
  }
  return -1;
}


cv::Mat findLargestContour(const cv::Mat &mask)
{
  cv::Mat result(mask.size(), CV_8UC1, cv::Scalar(0));

  std::vector<std::vector<cv::Point>> contours;

  cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  double maxArea = 0;
  int maxAreaIdx = -1;
  for (int i = 0; i < contours.size(); ++i)
  {
    double area = cv::contourArea(contours[i]);
    if (area > maxArea)
    {
      maxArea = area;
      maxAreaIdx = i;
    }
  }

  if (maxAreaIdx >= 0)
  {
    cv::drawContours(result, contours, maxAreaIdx, cv::Scalar(255), cv::FILLED);
  }


  return result;
}




int infer_rail::findEdge(cv::Mat & rail_mask, const int & far_flag){
  
  std::vector<cv::Point> left_points{};
  std::vector<cv::Point> right_points{};


  for(int row = rail_mask.rows -10 ; row > 0; row --){
    for(int col = 0; col < rail_mask.cols; col++){
      if(rail_mask.at<uint8_t>(row,col) == 255){
        left_points.push_back(cv::Point(col,row));
        break;
      }
    }
  }


  for(int row = rail_mask.rows -10 ; row > 0; row --){
    for(int col = rail_mask.cols; col > 0; col--){
      if(rail_mask.at<uint8_t>(row,col) == 255){
        right_points.push_back(cv::Point(col,row));
        break;
      }
    }
  }


    std::vector<std::vector<cv::Point>> left_split = splitRailPoints(left_points,1);
    std::vector<std::vector<cv::Point>> right_split = splitRailPoints(right_points,1);

    std::vector<bool> left_result = findAbnormalCluster(left_split,0, far_flag);
    std::vector<bool> right_result = findAbnormalCluster(right_split,1,far_flag);

    int max_left_pos = findMaxPosition(left_result);
    int max_right_pos = findMaxPosition(right_result);
    

    std::vector<cv::Point> left_result_cluster = left_split[max_left_pos];
    std::vector<cv::Point> right_result_cluster = right_split[max_right_pos];

    cv::Point left_result_point = left_result_cluster[0];
    cv::Point right_result_point = right_result_cluster[0];

    int lidar_distance = height_ - 1;

    // 远处lidar_distance投影到近处, 外部可视化
    
    // if(far_flag == 0){

    //     lidar_distance = std::max(left_result_point.y, right_result_point.y);
        
    //     p1_ = left_result_point;
    //     p2_ = right_result_point;

    // }else {

    //     std::vector<cv::Point2f> obj_corners{left_result_point, right_result_point};
    //     std::vector<cv::Point2f> scene_corners;
        
    //     perspectiveTransform( obj_corners, scene_corners, H_);

    //     lidar_distance = std::max(scene_corners[0].y, scene_corners[1].y);

    //     p1_origin_far_ = left_result_point;
    //     p2_origin_far_ = right_result_point;
    //     p1_far_ = scene_corners[0];
    //     p2_far_ = scene_corners[1];

    // }
    

    lidar_distance = std::max(left_result_point.y, right_result_point.y);

    return lidar_distance; 


}

 std::vector<std::vector<cv::Point>> infer_rail::splitRailPoints(std::vector<cv::Point> & pts, const int & len){

    std::vector<std::vector<cv::Point>> result_points{};

    for(int i = 0 ; i < pts.size() -len  + 1; i += len){
      
      std::vector<cv::Point> temp;
      for(int j = 0 ; j < len ; j++){
        temp.push_back(pts[i+j]);

      }
      result_points.push_back(temp);


    }
    return result_points;

}

std::vector<bool> infer_rail::findAbnormalCluster( std::vector<std::vector<cv::Point>> & splits, const int & flag, const int & far_flag){
      
      std::vector<bool> result;
      
      float k = 0;
      int x0 = 0;
      int y0 = 0;

      float k2 = 0;
      int x02 = 0;
      int y02 = 0;

      if(far_flag == 0){

          if(flag == 0){

              k = cal_line1_[1]/cal_line1_[0];
              x0 = cal_line1_[2];
              y0 = cal_line1_[3];

              k2 = cal_line2_[1]/cal_line2_[0];
              x02 = cal_line2_[2];
              y02 = cal_line2_[3];
          }else if(flag == 1){
              k = cal_line3_[1]/cal_line3_[0];
              x0 = cal_line3_[2];
              y0 = cal_line3_[3];

              k2 = cal_line4_[1]/cal_line4_[0];
              x02 = cal_line4_[2];
              y02 = cal_line4_[3];

          }

      }else if(far_flag==1){

          if(flag == 0){
              k = cal_line1_far_[1]/cal_line1_far_[0];
              x0 = cal_line1_far_[2];
              y0 = cal_line1_far_[3];

              k2 = cal_line2_far_[1]/cal_line2_far_[0];
              x02 = cal_line2_far_[2];
              y02 = cal_line2_far_[3];
          }else if(flag == 1){
              k = cal_line3_far_[1]/cal_line3_far_[0];
              x0 = cal_line3_far_[2];
              y0 = cal_line3_far_[3];

              k2 = cal_line4_far_[1]/cal_line4_far_[0];
              x02 = cal_line4_far_[2];
              y02 = cal_line4_far_[3];

          }

      }


      for(auto item : splits ){

        int cnt = 0;
        for(auto i : item){

          float x = ( i.y - y0)/k + x0;
          float x2 = (i.y - y02)/k2 + x02;

          if( x<=i.x && i.x<=x2){
              cnt++;

          }  
        }

        if(cnt>0){
          result.push_back(true);
        }else{
          result.push_back(false);
        }

      }
      return result;


}


int infer_rail::findMaxPosition(std::vector<bool> & temp){

    int longestStart = -1;
    int longestLength = 0;
    int currentStart = -1;
    int currentLength = 0;

    for (int i = 0; i <= temp.size(); ++i) {  // 注意这里是小于等于temp.size()
        if (i < temp.size() && temp[i] == 0) {  // 检查是否越界
            if (currentStart == -1) {
                currentStart = i;
            }
            currentLength++;
        } else {
            if (currentLength > longestLength) {
                longestStart = currentStart;
                longestLength = currentLength;
            }
            currentStart = -1;
            currentLength = 0;
        }
    }

    if (longestStart == -1) {  // 如果数组全为1
        longestStart = temp.size() - 1;
    }

    return longestStart;

}


void infer_rail::fitLines(const int & flag){

      std::vector<cv::Point>fit_points1;
      std::vector<cv::Point>fit_points2;
      std::vector<cv::Point>fit_points3;
      std::vector<cv::Point>fit_points4;

      if(flag == 0){
        
        fit_points1.push_back(cv::Point(p_left_top_.x     -  offset_top_,     p_left_top_.y)) ;
        fit_points1.push_back(cv::Point(p_left_bottom_.x  -  offset_bottom_,  p_left_bottom_.y));

        fit_points2.push_back(cv::Point(p_left_top_.x    +  offset_top_,       p_left_top_.y));
        fit_points2.push_back(cv::Point(p_left_bottom_.x +  offset_bottom_,  p_left_bottom_.y));

        fit_points3.push_back(cv::Point(p_right_top_.x     -  offset_top_,    p_right_top_.y)) ;
        fit_points3.push_back(cv::Point(p_right_bottom_.x  -  offset_bottom_, p_right_bottom_.y));
        
        fit_points4.push_back(cv::Point(p_right_top_.x    +  offset_top_,    p_right_top_.y));
        fit_points4.push_back(cv::Point(p_right_bottom_.x +  offset_bottom_, p_right_bottom_.y));


        cv::fitLine(fit_points1, line1_, cv::DIST_L2, 0, 1e-2, 1e-2);
        cv::fitLine(fit_points2, line2_, cv::DIST_L2, 0, 1e-2, 1e-2);
        cv::fitLine(fit_points3, line3_, cv::DIST_L2, 0, 1e-2, 1e-2);
        cv::fitLine(fit_points4, line4_, cv::DIST_L2, 0, 1e-2, 1e-2);

      }else if(flag == 1){

        fit_points1.push_back(cv::Point(p_left_top_far_.x     -  offset_top_far_,     p_left_top_far_.y)) ;
        fit_points1.push_back(cv::Point(p_left_bottom_far_.x  -  offset_bottom_far_,  p_left_bottom_far_.y));

        fit_points2.push_back(cv::Point(p_left_top_far_.x    +  offset_top_far_,     p_left_top_far_.y));
        fit_points2.push_back(cv::Point(p_left_bottom_far_.x +  offset_bottom_far_,  p_left_bottom_far_.y));

        fit_points3.push_back(cv::Point(p_right_top_far_.x     -  offset_top_far_,    p_right_top_far_.y)) ;
        fit_points3.push_back(cv::Point(p_right_bottom_far_.x  -  offset_bottom_far_, p_right_bottom_far_.y));
        
        fit_points4.push_back(cv::Point(p_right_top_far_.x    +  offset_top_far_,    p_right_top_far_.y));
        fit_points4.push_back(cv::Point(p_right_bottom_far_.x +  offset_bottom_far_, p_right_bottom_far_.y));

        cv::fitLine(fit_points1, line1_far_, cv::DIST_L2, 0, 1e-2, 1e-2);
        cv::fitLine(fit_points2, line2_far_, cv::DIST_L2, 0, 1e-2, 1e-2);
        cv::fitLine(fit_points3, line3_far_, cv::DIST_L2, 0, 1e-2, 1e-2);
        cv::fitLine(fit_points4, line4_far_, cv::DIST_L2, 0, 1e-2, 1e-2);

      }else if(flag == 2){
        
        fit_points1.push_back(cv::Point(back_p_left_top_.x     -  back_offset_top_,     back_p_left_top_.y)) ;
        fit_points1.push_back(cv::Point(back_p_left_bottom_.x  -  back_offset_bottom_,  back_p_left_bottom_.y));

        fit_points2.push_back(cv::Point(back_p_left_top_.x    +  back_offset_top_,       back_p_left_top_.y));
        fit_points2.push_back(cv::Point(back_p_left_bottom_.x +  back_offset_bottom_,    back_p_left_bottom_.y));

        fit_points3.push_back(cv::Point(back_p_right_top_.x     -  back_offset_top_,    back_p_right_top_.y)) ;
        fit_points3.push_back(cv::Point(back_p_right_bottom_.x  -  back_offset_bottom_, back_p_right_bottom_.y));
        
        fit_points4.push_back(cv::Point(back_p_right_top_.x    +  back_offset_top_,    back_p_right_top_.y));
        fit_points4.push_back(cv::Point(back_p_right_bottom_.x +  back_offset_bottom_, back_p_right_bottom_.y));


        cv::fitLine(fit_points1, back_line1_, cv::DIST_L2, 0, 1e-2, 1e-2);
        cv::fitLine(fit_points2, back_line2_, cv::DIST_L2, 0, 1e-2, 1e-2);
        cv::fitLine(fit_points3, back_line3_, cv::DIST_L2, 0, 1e-2, 1e-2);
        cv::fitLine(fit_points4, back_line4_, cv::DIST_L2, 0, 1e-2, 1e-2);

      }else if(flag==3){

        fit_points1.push_back(cv::Point(back_p_left_top_far_.x     -  back_offset_top_far_,     back_p_left_top_far_.y)) ;
        fit_points1.push_back(cv::Point(back_p_left_bottom_far_.x  -  back_offset_bottom_far_,  back_p_left_bottom_far_.y));

        fit_points2.push_back(cv::Point(back_p_left_top_far_.x    +  back_offset_top_far_,     back_p_left_top_far_.y));
        fit_points2.push_back(cv::Point(back_p_left_bottom_far_.x +  back_offset_bottom_far_,  back_p_left_bottom_far_.y));

        fit_points3.push_back(cv::Point(back_p_right_top_far_.x     -  back_offset_top_far_,    back_p_right_top_far_.y)) ;
        fit_points3.push_back(cv::Point(back_p_right_bottom_far_.x  -  back_offset_bottom_far_, back_p_right_bottom_far_.y));
        
        fit_points4.push_back(cv::Point(back_p_right_top_far_.x    +  back_offset_top_far_,    back_p_right_top_far_.y));
        fit_points4.push_back(cv::Point(back_p_right_bottom_far_.x +  back_offset_bottom_far_, back_p_right_bottom_far_.y));

        cv::fitLine(fit_points1, back_line1_far_, cv::DIST_L2, 0, 1e-2, 1e-2);
        cv::fitLine(fit_points2, back_line2_far_, cv::DIST_L2, 0, 1e-2, 1e-2);
        cv::fitLine(fit_points3, back_line3_far_, cv::DIST_L2, 0, 1e-2, 1e-2);
        cv::fitLine(fit_points4, back_line4_far_, cv::DIST_L2, 0, 1e-2, 1e-2);

      }

}



int infer_rail::nearRailNew(point_set & target, cv::Mat & rail_mask){
    
    if (target.xmin < 0){
        target.xmin = 0;
    }
    if (target.xmax > rail_mask.cols -1 ){
        target.xmax = rail_mask.cols -1;
    }
    if (target.ymin < 0){
        target.ymin = 0;
    }
    if (target.ymax > rail_mask.rows -1 ){
        target.ymax = rail_mask.rows -1;
    }

    cv::Rect rect(cv::Point(target.xmin, target.ymin), cv::Point(target.xmax, target.ymax));

    cv::Mat roi = rail_mask(rect);

    int nonzeroCount = cv::countNonZero(roi);

    if(nonzeroCount > 0){
        return 2;
    }
    else{
        return 1;
    }

   return 0;

}



void infer_rail::draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
 
    static const unsigned char colors[81][3] = {
        {56, 0, 255},
        {226, 255, 0},
        {0, 94, 255},
        {0, 37, 255},
        {0, 255, 94},
        {255, 226, 0},
        {0, 18, 255},
        {255, 151, 0},
        {170, 0, 255},
        {0, 255, 56},
        {255, 0, 75},
        {0, 75, 255},
        {0, 255, 169},
        {255, 0, 207},
        {75, 255, 0},
        {207, 0, 255},
        {37, 0, 255},
        {0, 207, 255},
        {94, 0, 255},
        {0, 255, 113},
        {255, 18, 0},
        {255, 0, 56},
        {18, 0, 255},
        {0, 255, 226},
        {170, 255, 0},
        {255, 0, 245},
        {151, 255, 0},
        {132, 255, 0},
        {75, 0, 255},
        {151, 0, 255},
        {0, 151, 255},
        {132, 0, 255},
        {0, 255, 245},
        {255, 132, 0},
        {226, 0, 255},
        {255, 37, 0},
        {207, 255, 0},
        {0, 255, 207},
        {94, 255, 0},
        {0, 226, 255},
        {56, 255, 0},
        {255, 94, 0},
        {255, 113, 0},
        {0, 132, 255},
        {255, 0, 132},
        {255, 170, 0},
        {255, 0, 188},
        {113, 255, 0},
        {245, 0, 255},
        {113, 0, 255},
        {255, 188, 0},
        {0, 113, 255},
        {255, 0, 0},
        {0, 56, 255},
        {255, 0, 113},
        {0, 255, 188},
        {255, 0, 94},
        {255, 0, 18},
        {18, 255, 0},
        {0, 255, 132},
        {0, 188, 255},
        {0, 245, 255},
        {0, 169, 255},
        {37, 255, 0},
        {255, 0, 151},
        {188, 0, 255},
        {0, 255, 37},
        {0, 255, 0},
        {255, 0, 170},
        {255, 0, 37},
        {255, 75, 0},
        {0, 0, 255},
        {255, 207, 0},
        {255, 0, 226},
        {255, 245, 0},
        {188, 255, 0},
        {0, 255, 18},
        {0, 255, 75},
        {0, 255, 151},
        {255, 56, 0},
        {245, 255, 0}
    };
    
    cv::Mat image = bgr.clone();

    int color_index = 0;

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        if (obj.prob < 0.15)
            continue;

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        const unsigned char* color = colors[color_index % 81];
        color_index++;

        cv::rectangle(image, obj.rect, cv::Scalar(color[0], color[1], color[2]));

        char text[256];
        sprintf(text, "%s %.1f%%", obj.label.c_str(), obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    // draw mask
        for (int y = 0; y < image.rows; y++)
        {
            const uchar* mp = obj.mask.ptr(y);
            uchar* p = image.ptr(y);
            for (int x = 0; x < image.cols; x++)
            {
                if (mp[x] == 255)
                {
                    p[0] = cv::saturate_cast<uchar>(p[0] * 0.5 + color[0] * 0.5);
                    p[1] = cv::saturate_cast<uchar>(p[1] * 0.5 + color[1] * 0.5);
                    p[2] = cv::saturate_cast<uchar>(p[2] * 0.5 + color[2] * 0.5);
                }
                p += 3;
            }
        }
    }
    imwrite("../testout.jpg",image);
}



std::vector<Configuration> infer_rail::readConfig(std::string & configPath){
    
    std::vector<Configuration> configurations;

    std::ifstream configFile;
    configFile.open(configPath);
    if (!configFile.is_open()) {
        std::cerr << "Error opening config file." << std::endl;
        return configurations;
    }

   
    Configuration currentConfig;
    std::string line;
    while (std::getline(configFile, line)) {
        if (line.empty()) continue; // 跳过空行

        if (line[0] == '[') { // 新的大类开始
            if (!currentConfig.data.empty()) { // 如果之前的大类有配置数据，存储起来
                configurations.push_back(currentConfig);
                currentConfig.data.clear();
            }
            // 保存大类名

            line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
            currentConfig.section = line;
        } else if (line[0] != '%') { // 不是注释行
            // Split line into key and value
            std::istringstream iss(line);
            std::string key, value;
            if (std::getline(iss, key, '=')) {
                if (std::getline(iss, value)) {
                    // Remove leading and trailing whitespace from key and value
                    key.erase(0, key.find_first_not_of(" \t"));
                    key.erase(key.find_last_not_of(" \t") + 1);
                    value.erase(0, value.find_first_not_of(" \t"));
                    value.erase(value.find_last_not_of(" \t") + 1);

                  
                    key.erase(std::remove(key.begin(), key.end(), '\r'), key.end());
                    value.erase(std::remove(value.begin(), value.end(), '\r'), value.end());
                    currentConfig.data[key] = value;
                
                
                }
            }
        }
    }
    // Store the last configuration
    if (!currentConfig.data.empty()) {
        configurations.push_back(currentConfig);
    }



    return configurations;

}

