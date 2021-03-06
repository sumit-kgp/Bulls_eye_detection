#include "non_maxima_suppression2.h"
#include "multiple_tracking.h"
#include "least_square_detection.h"
#include "mainwindow.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>
#include "calcmaxintensity.h"
#include "opencv2/highgui/highgui.hpp"
#include "algorithm"

using namespace cv;
using namespace std;

//structure to store pixels values and corresponding coordinates
struct sorted_pixels{
    Point sorted_pixels;
    double intensity;
};

//function to sort coordinates based on intensity
bool wayToSort(const sorted_pixels &a, const sorted_pixels &b) {return a.intensity>b.intensity;}

non_maxima_suppression2::non_maxima_suppression2(cv::Mat mask2, std::vector<cv::Point> *p_temp, int background_buffer_counter)
{
    char file[125];
int blocksize = 50;
int top, bottom, left, right;
const int M = mask2.rows;
const int N = mask2.cols;
Point minLoc, maxLoc;
double maxVal;
Mat nms_mask = Mat::zeros(mask2.size(), mask2.type());
top = (int) (blocksize); bottom = (int) (blocksize);
left = (int) (blocksize); right = (int) (blocksize);
copyMakeBorder( mask2, mask2, top, bottom, left, right, BORDER_CONSTANT, 0 );
vector<sorted_pixels> intensity_sort;
sorted_pixels element;


    for(int i=0; i<M+1; i+=blocksize)
    {
        for(int j=0; j<N+1; j+=blocksize)
            {

            Mat mask2ROI(mask2,cv::Rect(j,i,blocksize,blocksize));
            minMaxLoc(mask2ROI, NULL, &maxVal, &minLoc, &maxLoc);
                if(maxVal>0){
                    //Translating the coordinates of local maxima to cartesian ones of the mask2
                    maxLoc.x=maxLoc.x+j-blocksize;
                    maxLoc.y=maxLoc.y+i-blocksize;

                element.sorted_pixels = maxLoc;
                element.intensity = maxVal;
                intensity_sort.push_back(element);

                nms_mask.at<uchar>(maxLoc) = maxVal;

                //circle(nms_mask, maxLoc, 0.1*maxVal, Scalar(255,0,255), 3, 8, 0);

                //line(nms_mask,Point(maxLoc.x-0.1*maxVal, maxLoc.y),Point(maxLoc.x+0.1*maxVal, maxLoc.y),Scalar(255,255,255),0.1,8,0);
                //line(nms_mask,Point(maxLoc.x, maxLoc.y-0.1*maxVal),Point(maxLoc.x, maxLoc.y+0.1*maxVal),Scalar(255,255,255),0.1,8,0);

                }

        }

    }

 std::sort(intensity_sort.begin(), intensity_sort.end(), wayToSort);
vector<sorted_pixels>::iterator it;
         for(it= intensity_sort.begin(); it!=intensity_sort.end(); it++)
         {
            p_temp->push_back(it->sorted_pixels);

         }

 imshow("after non maximal suppression", nms_mask);
 //sprintf(file,"/home/sumit/nms/NMS_%d.png\n", background_buffer_counter);
 //imwrite(file, nms_mask);
 //imwrite("/home/sumit/Desktop/tracking image/non_max_supp2.jpg", nms_mask);

}

