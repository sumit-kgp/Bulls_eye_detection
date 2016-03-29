#include "calcmaxintensity.h"
#include "mainwindow.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>
#include <cmath>
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

calcMaxIntensity::calcMaxIntensity(Mat src_gray, Mat grad_x, Mat grad_y, Mat* grad)
{
    int resolution = 1;

    for (int i = 0 ; i < src_gray.rows ; i+= resolution)
    {

        float* pixel_x = grad_x.ptr<float>(i);
        float* pixel_y = grad_y.ptr<float>(i);
        float* pixel = grad->ptr<float>(i);

                for (int j = 0 ; j < src_gray.cols ; j+= resolution)
                {
                    *pixel = std::sqrt(std::pow(*pixel_x,2)+std::pow(*pixel_y,2));//5000.0;    //100 for resolution 5 and 100000 for res 1

                     pixel_x = pixel_x+resolution;
                     pixel_y = pixel_y+resolution;
                     pixel = pixel+resolution;
                }

    }

}

