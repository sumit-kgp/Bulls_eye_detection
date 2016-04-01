#include "non_maxima_suppression.h"
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

non_maxima_suppression::non_maxima_suppression(Mat* mask2, vector<Point>* p_temp)
{
    //FIND GLOBAL MAXIMA

    Point minLoc, maxLoc;
    minMaxLoc(*mask2, NULL, NULL, &minLoc, &maxLoc);
    p_temp->push_back(maxLoc);

    //EXTRACTING ROI AROUND MAXLOC

    //Point newLoc;
    float offset = 30.0;
    //newLoc.x=offset;
    //newLoc.y=offset;
    Mat mask2ROI(*mask2,cv::Rect(maxLoc.x-offset,maxLoc.y-offset,2*offset,2*offset));
    mask2ROI = Mat::zeros(mask2ROI.size(), mask2ROI.type());
    imshow("after suppression", *mask2);
}

