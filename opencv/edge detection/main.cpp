//https://www.cnblogs.com/frombeijingwithlove/p/4226489.html
//https://www.zhihu.com/question/27042602/answer/37301260

//(962,1104),(1704,1120)
//(1213,2539),(2208,2418)
#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat image = cv::imread("src.jpg");

    // 目标边缘的四个顶点坐标
    std::vector<cv::Point2f> srcPoints = {
        cv::Point2f(962,1104),  // 左上
        cv::Point2f(1704,1200),  // 右上
        cv::Point2f(2180,2480),  // 右下
        cv::Point2f(1213,2550)   // 左下
    };

    // 透视变换后的目标四个顶点坐标
    std::vector<cv::Point2f> dstPoints = {
        cv::Point2f(0, 0),                       // 左上
        cv::Point2f(image.cols, 0),              // 右上
        cv::Point2f(image.cols, image.rows),     // 右下
        cv::Point2f(0, image.rows)               // 左下
    };


    // 透视变换
    cv::Mat perspectiveMatrix = cv::getPerspectiveTransform(srcPoints, dstPoints);

    cv::Mat warped;
    cv::warpPerspective(image, warped, perspectiveMatrix, image.size());
   
    //变换成17/9大小
    int width = 360;  
    int height = 680;  

    cv::Mat resized;  

    cv::resize(warped, resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);  

    cv::imshow("Warped Image", resized);
    cv::imwrite("warped_image.jpg", resized);
    cv::waitKey(0);

    return 0;
}