#ifndef SUPPRESS_LOCAL_NEIGHBORHOOD_H
#define SUPPRESS_LOCAL_NEIGHBORHOOD_H
#include "mainwindow.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>
#include "calcmaxintensity.h"

class suppress_local_neighborhood
{
public:
    suppress_local_neighborhood(Mat mask, Mat *mask2);
};

#endif // SUPPRESS_LOCAL_NEIGHBORHOOD_H
