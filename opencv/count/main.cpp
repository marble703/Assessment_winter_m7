#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat image = cv::imread("count.png");
    if (image.empty()) {
        std::cerr << "Failed to read image." << std::endl;
        return -1;
    }

    int count = 0; 

    // 灰度图
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    int kernel_size = 5;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
    // 腐蚀
    cv::Mat eroded_image;
    cv::erode(gray, eroded_image, kernel);

    // 对图像进行边缘检测
    cv::Mat edges;
    cv::Canny(eroded_image, edges, 50, 200);

    // 寻找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 绘制轮廓和矩形框  
    cv::Mat annotatedImage = edges.clone(); // 使用原图作为背景绘制  
    for (size_t i = 0; i < contours.size(); i++) {  

        // 计算最小矩形框  
        cv::Rect boundingRect = cv::boundingRect(contours[i]);  

        // 检查矩形框的大小是否满足要求
        if (boundingRect.width >= 150 && boundingRect.height >= 150) {

            count += 1;

            // 绘制矩形框  
            cv::rectangle(annotatedImage, boundingRect, cv::Scalar(255, 255, 255), 2);  
        
            // 在矩形框中心书写编号
            std::string text = std::to_string(count);  
            int fontFace = cv::FONT_HERSHEY_SIMPLEX;  
            double fontScale = 1;  
            int thickness = 2;  
            cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, nullptr);
            cv::Point textOrg(boundingRect.x + (boundingRect.width - textSize.width) / 2, 
                              boundingRect.y + (boundingRect.height + textSize.height) / 2);  
            cv::putText(annotatedImage, text, textOrg, fontFace, fontScale, cv::Scalar(255, 255, 255), thickness);  
        }
    }  

    std::cout << "Number of objects detected: " << count << std::endl;

    // 显示结果
    cv::imshow("Contours", annotatedImage);
    cv::imwrite("Contours.jpg", annotatedImage);
    cv::waitKey(0);

    return 0;
}
