#include "multiple_tracking.h"
#include "least_square_detection.h"
#include "mainwindow.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>
#include "calcmaxintensity.h"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

multiple_tracking::multiple_tracking(Mat mask, Mat dst, double maxVal)
{
    int resolution = 1;
    Point temp(0,0);
    //if to detect multiple centres based on thresholding
    for(int i = 0 ; i < mask.rows ; i+= resolution){
        float* pixel = mask.ptr<float>(i);
        for(int j = 0 ; j < mask.cols ; j+= resolution)
        {
            if(mask.at<uchar>(i,j)>0.6*maxVal){
                Point p(j,i);
                float dist = std::sqrt(std::pow(temp.x-p.x,2)+std::pow(temp.y-p.y,2));
                 if(dist>5){
                    circle(dst, p, 1, Scalar(255,0,0), 1, 8, 0);
                    std::cout << "points" << std::endl << p.x << "," <<p.y<< std::endl;
                    temp = p;
                 }
                }
        }
        pixel = pixel+resolution;
    }
    imshow("equalized_pre", dst);
}
