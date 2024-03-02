#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main()
{
    Mat image1, output_image, image1_gray;
    image1 = imread("src.jpg");

    cvtColor(image1, image1_gray, COLOR_BGR2GRAY);
    imshow("image1_gray", image1_gray);

    output_image = image1_gray.clone();
    double gamma =2;
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
 
    cv::imshow("gamma.jpg", output_image);
    cv::imwrite("gamma.jpg", output_image);

    waitKey(0);
    return 0;
}
