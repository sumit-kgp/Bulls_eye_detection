#ifndef MULTIPLE_TRACKING_H
#define MULTIPLE_TRACKING_H
#include "mainwindow.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>
#include "calcmaxintensity.h"
//#include <QQuickItem>

class multiple_tracking
{
public:
   multiple_tracking(cv::Mat mask, cv::Mat dst, double maxVal, std::vector<cv::Point> *p_temp );
};

#endif // MULTIPLE_TRACKING_H
