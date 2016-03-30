#ifndef ADDSTRAIGHTLINE_H
#define ADDSTRAIGHTLINE_H
#include "mainwindow.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>

//#include <QQuickItem>

class addStraightLine
{
public:
    addStraightLine(cv::Mat *mask, cv::Point2f p1, cv::Point2f p2, cv::Scalar color);
};

#endif // ADDSTRAIGHTLINE_H
