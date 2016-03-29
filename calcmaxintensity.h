#ifndef CALCMAXINTENSITY_H
#define CALCMAXINTENSITY_H
#include "mainwindow.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>

//#include <QQuickItem>
using namespace cv;

class calcMaxIntensity
{
public:
    calcMaxIntensity(Mat src_gray, Mat grad_x, Mat grad_y, Mat* grad);
};

#endif // CALCMAXINTENSITY_H
