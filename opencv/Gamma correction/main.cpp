#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void gamma_correct(const cv::Mat& img, cv::Mat& output_image){
    cv::Mat  image1_gray;
    cvtColor(img, image1_gray, COLOR_BGR2GRAY);

    output_image = image1_gray.clone();
    double gamma = 0.5;
    for (int i = 0; i < image1_gray.rows; i++)
    {
        for (int j = 0; j < image1_gray.cols; j++)
        {
            // 伽马矫正
            double pixelValue = static_cast<double>(image1_gray.at<uchar>(i,j));
            double correctedPixelValue = pow(pixelValue / 255, gamma) * 255;
            output_image.at<uchar>(i, j) = static_cast<uchar>(correctedPixelValue);
        }
    }
}

int main()
{
    Mat image1, image2, output_image1, output_image2;
    image1 = imread("1.png");
    image2 = imread("2.jpg");

    gamma_correct(image1, output_image1);
    gamma_correct(image2, output_image2);
 
    cv::imshow("gamma1.jpg", output_image1);
    cv::imshow("gamma2.jpg", output_image2);

    cv::imwrite("gamma1.jpg", output_image1);
    cv::imwrite("gamma2.jpg", output_image2);
    waitKey(0);
    return 0;
}
