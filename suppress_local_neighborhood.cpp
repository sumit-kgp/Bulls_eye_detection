#include "suppress_local_neighborhood.h"
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

suppress_local_neighborhood::suppress_local_neighborhood(cv::Mat mask, cv::Mat* mask2)
{
    int erosion_size = 3;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                          cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                          cv::Point(erosion_size, erosion_size) );
    // find pixels that are equal to the local neighborhood not maximum (including 'plateaus')
       cv::dilate(mask, *mask2, element);
       imshow("dilate", *mask2);
       cv::compare(mask, *mask2, *mask2, cv::CMP_GE);
       imshow("1st compare", *mask2);

           // optionally filter out pixels that are equal to the local minimum ('plateaus')
           //cv::Mat non_plateau_mask;
           //cv::erode(mask, non_plateau_mask, cv::Mat());
           //imshow("erode", non_plateau_mask);
           //cv::compare(mask, non_plateau_mask, non_plateau_mask, cv::CMP_GT);
           //imshow("2nd compare", non_plateau_mask);
           //cv::bitwise_and(mask2, non_plateau_mask, mask2);
           //imshow("bitwise and", mask2);

       cv::bitwise_and(*mask2, mask, *mask2);
}

