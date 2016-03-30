#include "least_square_detection.h"
#include "mainwindow.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>
#include "calcmaxintensity.h"

using namespace cv;
using namespace std;

least_square_detection::least_square_detection(cv::Mat* srcROI, cv::Mat grad_x, cv::Mat grad_y, cv::Mat grad, cv::Point newLoc, float* X, float* Y)
{
    Mat_<float> final_product;
    Mat_<float> num_buf;
    Mat_<float> num_buf1;
    Mat_<float> den_buf;
    Mat_<float> den_buf1;
    Mat_<float> V_buf_t(2,2);
    Mat I = (Mat_<float>(2,2) << 1, 0, 0, 1);

    int resolution = 1;

    calcMaxIntensity(*srcROI, grad_x, grad_y, &grad);
    double minVal, maxVal;
    minMaxLoc(grad, &minVal, &maxVal, NULL, NULL);

    for (int i = 0 ; i < srcROI->rows ; i+= resolution)
    {

         for (int j = 0 ; j < srcROI->cols ; j+= resolution)
        {

            float* pixel_x = grad_x.ptr<float>(i,j);
            float* pixel_y = grad_y.ptr<float>(i,j);
            float* pixel = grad.ptr<float>(i,j);

        Point2f p(j,i),p2;
        Point2f grad_ptr;
        grad_ptr.x = *pixel_x;
        grad_ptr.y = *pixel_y;


                if(*pixel>0.04*maxVal)
                {
                p2.x = p.x+0.01*grad_ptr.x;
                p2.y = p.y+0.01*grad_ptr.y;
                float per_dist;

                        // Slope equation (y1 - y2) / (x1 - x2)
                        if (p.x != p2.x)
                        {
                        float m = (p.y - p2.y) / (p.x - p2.x);
                        // Line equation:  y = mx + b
                        float b = p.y - (m * p.x);
                        per_dist = std::abs((newLoc.y-m*newLoc.x-b)/(std::sqrt(1+std::pow(m,2))));
                        }
                        else
                        {
                        per_dist = std::abs(newLoc.x-p.x);
                        }
                        float distance = std::sqrt(std::pow(p.x-newLoc.x,2)+std::pow(p.y-newLoc.y,2));


                //if the line passes very close to the centre, then compute least square coordinate


                                      if(  (per_dist<3)&&(distance<100))
                                        {



                                            Mat_<float> V_buf(1,1);
                                            Mat V = (Mat_<float>(2,1) << (p2.x-p.x), (p2.y-p.y));


                                            mulTransposed(V, V_buf, true);
                                            mulTransposed(V, V_buf_t, false);

                                            Mat_<float> P(2,1);
                                            P << p.x, p.y;

                                            subtract(I,(V_buf_t)*(1/V_buf[0][0]), den_buf1);
                                            num_buf1 = den_buf1*P;

                                            num_buf = num_buf1+num_buf;
                                            den_buf = den_buf1+den_buf;


                                           arrowedLine(*srcROI,p2,p,Scalar(0,255,0),0.1,8,0,0.1);

                                        }
                }
        pixel_x = pixel_x+resolution;
        pixel_y = pixel_y+resolution;
        pixel = pixel+resolution;
        }
    }


   den_buf = den_buf.inv();
   final_product = den_buf*num_buf;
   *X = final_product.at<float>(0,0);
   *Y = final_product.at<float>(1,0);

}

