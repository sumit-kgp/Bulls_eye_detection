#ifndef LEAST_SQUARE_DETECTION2_H
#define LEAST_SQUARE_DETECTION2_H
#include "mainwindow.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>
#include "calcmaxintensity.h"

class least_square_detection2
{
public:
    least_square_detection2(cv::Mat *srcROI, cv::Mat grad_x, cv::Mat grad_y, cv::Mat grad, cv::Point newLoc, float *X, float *Y);
};

#endif // LEAST_SQUARE_DETECTION2_H
