#ifndef NON_MAXIMA_SUPPRESSION_H
#define NON_MAXIMA_SUPPRESSION_H
#include "mainwindow.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>
#include "calcmaxintensity.h"

class non_maxima_suppression
{
public:
    non_maxima_suppression(cv::Mat *mask2, std::vector<cv::Point> *p_temp);
};

#endif // NON_MAXIMA_SUPPRESSION_H
