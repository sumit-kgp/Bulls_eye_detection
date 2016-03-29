#include "addstraightline.h"
#include "mainwindow.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>

using namespace cv;
using namespace std;


addStraightLine::addStraightLine(cv::Mat *mask, cv::Point2f p1, cv::Point2f p2, cv::Scalar color)
{
        Point2f p, q;
    // Check if the line is a vertical line because vertical lines don't have slope
        if (p1.x != p2.x)
        {
                p.x = 0;
                q.x = mask->cols;
                // Slope equation (y1 - y2) / (x1 - x2)
                float m = (p1.y - p2.y) / (p1.x - p2.x);
                // Line equation:  y = mx + b
                float b = p1.y - (m * p1.x);
                p.y = m * p.x + b;
                q.y = m * q.x + b;
        }
        else
        {
                p.x = q.x = p2.x;
                p.y = 0;
                q.y = mask->rows;
        }
        LineIterator it(*mask, p, q, 8);
        for(int i = 0; i < it.count; i++, ++it)
        {
            *(float*)(*it) += color[0];
        }

}
