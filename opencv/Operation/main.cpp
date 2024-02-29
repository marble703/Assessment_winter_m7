#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
    cv::Mat src = cv::imread("src.jpg", cv::IMREAD_GRAYSCALE);

    // 定义核
    int kernel_size = 5;

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));

    // 腐蚀
    cv::Mat eroded_image;
    cv::erode(src, eroded_image, kernel);

    // 膨胀
    cv::Mat dilated_image;
    cv::dilate(src, dilated_image, kernel);

    // 开运算
    cv::Mat opened_image;
    cv::morphologyEx(src, opened_image, cv::MORPH_OPEN, kernel);

    // 闭运算
    cv::Mat closed_image;
    cv::morphologyEx(src, closed_image, cv::MORPH_CLOSE, kernel);

    cv::imshow("Original Image.jpg", src);
    cv::imshow("Eroded Image.jpg", eroded_image);
    cv::imshow("Dilated Image.jpg", dilated_image);
    cv::imshow("Opened Image.jpg", opened_image);
    cv::imshow("Closed Image.jpg", closed_image);


    cv::imwrite("Original Image.jpg", src);
    cv::imwrite("Eroded Image.jpg", eroded_image);
    cv::imwrite("Dilated Image.jpg", dilated_image);
    cv::imwrite("Opened Image.jpg", opened_image);
    cv::imwrite("Closed Image.jpg", closed_image);

    cv::waitKey(0);
    return 0;
}
