#include <iostream>  
#include <opencv2/opencv.hpp>  
  
int main() {  
    // 读取图像  
    cv::Mat image = cv::imread("count.png");  
    if (image.empty()) {  
        std::cerr << "Failed to read image." << std::endl;  
        return -1;  
    }  

    // 将图像转换为灰度图  
    cv::Mat gray;  
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);  
  
    int kernel_size = 9;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
    // 腐蚀
    cv::Mat eroded_image;
    cv::erode(gray, eroded_image, kernel);

    cv::imshow("eroded_image", eroded_image); 

    // 进行连通域检测  
    cv::Mat labels, stats, centroids;  
    int num_labels = cv::connectedComponentsWithStats(eroded_image, labels, stats, centroids);  

    // 设置最小矩形框大小（以像素为单位）  
    int min_width = 0;
    int min_height = 0;

     // 绘制边界框  
    cv::Mat result = image.clone();  
    for (int i = 1; i < num_labels; i++) { // 跳过背景标签  
        // 获取当前连通域的统计信息  
        int left = stats.at<int>(i, cv::CC_STAT_LEFT);  
        int top = stats.at<int>(i, cv::CC_STAT_TOP);  
        int width = stats.at<int>(i, cv::CC_STAT_WIDTH);  
        int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);  

        // 检查连通域大小  
        if (width >= min_width && height >= min_height) {    
            cv::rectangle(result, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(0, 255, 0), 2);  
        }  
    }  

    std::cout<<num_labels - 1<<std::endl;
 
    cv::imshow("Result", result);  
    cv::imwrite("result.jpg", result);  
    cv::waitKey(0);  
  
    return 0;  
}