#ifndef NON_MAXIMA_SUPPRESSION2_H
#define NON_MAXIMA_SUPPRESSION2_H
#include "mainwindow.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>
#include "calcmaxintensity.h"
#include "algorithm"

class non_maxima_suppression2
{
public:
    non_maxima_suppression2(cv::Mat mask2, std::vector<cv::Point> *p_temp);
};

#endif // NON_MAXIMA_SUPPRESSION2_H
