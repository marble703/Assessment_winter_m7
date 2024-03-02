#include <iostream>
#include <opencv2/opencv.hpp>

int main() {

    cv::VideoCapture cap(0);

    cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);

    while (true) {
        cv::Mat frame;
        cap >> frame;

        cv::imshow("Camera", frame);

        char key = cv::waitKey(10);
        if (key == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
