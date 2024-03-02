#include<iostream>
#include<opencv2/opencv.hpp>

int main()
{
    cv::Mat src = cv::imread("1.png", cv::IMREAD_GRAYSCALE);

    cv::Mat binary;
    double threshold = 128;
    cv::threshold(src, binary, threshold, 255, cv::THRESH_BINARY);

    cv::imshow("Original", src);
    cv::imshow("Binarization", binary);
    cv::imwrite("binary_image.png", binary);

    cv::waitKey(0);

    return 0;
}
