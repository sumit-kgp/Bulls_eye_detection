#ifndef LEASTSQAUREDETECTION_H
#define LEASTSQAUREDETECTION_H
#include "mainwindow.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
//#include <stdlib.h>
//#include <QQuickItem>

class least_square_detection
{
public:
     least_square_detection(cv::Mat *srcROI, cv::Mat grad_x, cv::Mat grad_y, cv::Mat grad, cv::Point newLoc, float *X, float *Y);
};

#endif // LEASTSQAUREDETECTION_H
