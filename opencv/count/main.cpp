#include <iostream>
#include <opencv2/opencv.hpp>

/*
//删除小面积的连通域
cv::Mat deleteMinWhiteArea(cv::Mat src,int min_area) {
    cv::Mat labels, stats, centroids, img_color, grayImg;
    //1、连通域信息统计
    int nccomps = connectedComponentsWithStats(
        src, //二值图像
        labels,
        stats,
        centroids
    );

    //2、连通域状态区分
    //为每一个连通域初始化颜色表
    std::vector<Vec3b> colors(nccomps);
    colors[0] = Vec3b(0, 0, 0); // background pixels remain black.
    for (int i = 1; i < nccomps; i++)
    {
        colors[i] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
        //面积阈值筛选
        if ((stats.at<int>(i, cv::CC_STAT_AREA) < min_area))
        {
            //如果连通域面积不合格则置黑
            colors[i] = Vec3b(0, 0, 0);
        }
    }
    //3、连通域删除
    //按照label值，对不同的连通域进行着色
    img_color = cv::Mat::zeros(src.size(), CV_8UC3);
    for (int y = 0; y < img_color.rows; y++)
    {
        int* labels_p = labels.ptr<int>(y);//使用行指针，加速运算
        Vec3b* img_color_p = img_color.ptr<Vec3b>(y);//使用行指针，加速运算
        for (int x = 0; x < img_color.cols; x++)
        {
            int label = labels_p[x];//取出label值
            CV_Assert(0 <= label && label <= nccomps);
            img_color_p[x] = colors[label];//设置颜色
        }
    }
    //return img_color;
    //如果是需要二值结果则将img_color进行二值化
    cvtColor(img_color, grayImg, cv::COLOR_BGR2GRAY);
    threshold(grayImg, grayImg, 1, 255, cv::THRESH_BINARY);
    return grayImg;
}
*/
int main() {
    cv::Mat image = cv::imread("count.png");
    if (image.empty()) {
        std::cerr << "Failed to read image." << std::endl;
        return -1;
    }

    // 灰度图
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // 二值化
    cv::threshold(gray, gray, 128, 255, cv::THRESH_BINARY);

    int kernel_size = 13;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
    // 腐蚀
    cv::Mat eroded_image;
    cv::erode(gray, eroded_image, kernel);

    cv::imshow("eroded_image", eroded_image); 

    // 进行连通域检测
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(eroded_image, labels, stats, centroids);

    int min_size = 1000;
    int max_size = 742*737;
    int count = 0;

    // 绘制目标边界框
    cv::Mat result = image.clone();
    for (int i = 1; i < num_labels; ++i) { // 跳过背景标签
        // 获取当前连通域的统计信息
        int left = stats.at<int>(i, cv::CC_STAT_LEFT);
        int top = stats.at<int>(i, cv::CC_STAT_TOP);
        int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
 
        // 检查连通域大小  
        if (width * height >= min_size && width * height < max_size) {    
            cv::rectangle(result, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(0, 255, 0), 2);  
            count +=1;
        } 

    }

    std::cout<< count <<std::endl;
    cv::imshow("Result", result);
    cv::imwrite("result.png", result);
    cv::waitKey(0);

    return 0;
}
