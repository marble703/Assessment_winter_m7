#include<iostream>
#include<opencv2/opencv.hpp>

int main()
{
    // 读取图像
    cv::Mat src = cv::imread("1.png", cv::IMREAD_GRAYSCALE);

    // 二值化图像
    cv::Mat binary;
    double threshold = 128; // 设置阈值
    cv::threshold(src, binary, threshold, 255, cv::THRESH_BINARY);

    // 显示原始图像和二值化后的图像
    cv::imshow("Original Image", src);
    cv::imshow("Binary Image", binary);

    // 保存二值化图像
    cv::imwrite("binary_image.png", binary);
    
    // 等待按键，然后关闭窗口
    cv::waitKey(0);
    return 0;
}
