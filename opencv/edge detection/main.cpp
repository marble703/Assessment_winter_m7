/*
尽力了，为什么效果这么差
*/

#include <iostream>
#include <opencv2/opencv.hpp>

// 伽马矫正函数  
cv::Mat gammaCorrection(const cv::Mat& src) {  
    cv::Mat dst;  
    dst.create(src.size(), src.type());  

    cv::Mat floatSrc;  
    src.convertTo(floatSrc, CV_32F);  

    cv::pow(floatSrc, 2, dst);  

    cv::Mat normalizedDst;  
    cv::normalize(dst, normalizedDst, 0, 255, cv::NORM_MINMAX, CV_8U);  
  
    return normalizedDst;  
}  

int main()
{
    cv::Mat src = cv::imread("src.jpg", cv::IMREAD_GRAYSCALE);

    // 伽马矫正  
    cv::Mat gammaCorrected = gammaCorrection(src);  

    // 高斯滤波
    cv::Mat blurred;
    cv::GaussianBlur(gammaCorrected, blurred, cv::Size(3, 3), 0);

    // Canny
    double low_threshold = 150;
    double high_threshold = 200;
    cv::Mat edges;
    cv::Canny(blurred, edges, low_threshold, high_threshold);

   // 腐蚀
    cv::Mat dilated;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(edges, dilated, element);
    
    //膨胀
    cv::Mat dilated_image;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::dilate(src, dilated_image, kernel);

    cv::imshow("edge image", edges);
    cv::imwrite("edges_image.png", edges);

    cv::waitKey(0);
    return 0;
}