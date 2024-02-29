#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat rgb_image = cv::imread("src.jpg");

    cv::Mat hsv_image;
    cv::cvtColor(rgb_image, hsv_image, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv_image, hsv_channels);

    hsv_channels[0] += 30;

    hsv_channels[1] -= 30;

    hsv_channels[2] += 50;

    cv::Mat adjusted_hsv_image;
    cv::merge(hsv_channels, adjusted_hsv_image);

    
    cv::imshow("hsv_image.jpg", adjusted_hsv_image);
    cv::imwrite("hsv_image.jpg", adjusted_hsv_image);
    waitKey(0);
    
    return 0;
}
