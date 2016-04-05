#include "least_square_detection2.h"
#include "mainwindow.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>
#include "calcmaxintensity.h"

using namespace cv;
using namespace std;

least_square_detection2::least_square_detection2(cv::Mat* srcROI, cv::Mat grad_x, cv::Mat grad_y, cv::Mat grad, cv::Point newLoc, float* X, float* Y)
{
    Mat_<float> final_product;
    Mat_<float> num_buf(2,1);
    //Mat_<float> num_buf1;
    Mat_<float> den_buf(2,2);
    //float num_buf[2][1];
    float num_buf1[2][1] = {0,0};
    //float den_buf[2][2];
    //float den_buf [2][2];
    //Mat_<float> den_buf1;
    float den_buf1[2][2] = {0,0,0,0};
    //Mat_<float> V_buf_t(2,2);
    //Mat_<float> perpendicular;
    float perpendicular[2][1];
    float num_buf2[2][1]={0,0};
    float den_buf2[2][2];
    //Mat I = (Mat_<float>(2,2) << 1, 0, 0, 1);


    int resolution = 1;

    //calcMaxIntensity(*srcROI, grad_x, grad_y, &grad);

    for (int i = 0 ; i < srcROI->rows ; i+= resolution)
    {

         for (int j = 0 ; j < srcROI->cols ; j+= resolution)
        {

            float* pixel_x = grad_x.ptr<float>(i,j);
            float* pixel_y = grad_y.ptr<float>(i,j);
            float* pixel = grad.ptr<float>(i,j);
             *pixel = std::sqrt(std::pow(*pixel_x,2)+std::pow(*pixel_y,2));
            pixel_x = pixel_x+resolution;
            pixel_y = pixel_y+resolution;
            pixel = pixel+resolution;
        }
    }

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

            //double 2x2 m;
            //Mat V(m);
            //m
            float P[2][1]={p.x, p.y};
            float V[2][1] = { (p2.x-p.x), (p2.y-p.y) };
            //Mat V = (Mat_<float>(2,1) << (p2.x-p.x), (p2.y-p.y));
            float V_buf = (p2.x-p.x)*(p2.x-p.x)+(p2.y-p.y)*(p2.y-p.y);//std::pow(V.at<float>(0,0),2)+std::pow(V.at<float>(1,0),2);
            //mulTransposed(V, V_buf_t, false);
            float V_buf_t[2][2] = {(p2.x-p.x)*(p2.x-p.x), (p2.x-p.x)*(p2.y-p.y), (p2.x-p.x)*(p2.y-p.y), (p2.y-p.y)*(p2.y-p.y)};
            /*V_buf_t.at<float>(0,0)=(p2.x-p.x)*(p2.x-p.x);
            V_buf_t.at<float>(0,1)=(p2.x-p.x)*(p2.y-p.y);
            V_buf_t.at<float>(1,1)=(p2.y-p.y)*(p2.y-p.y);
            V_buf_t.at<float>(1,0)=(p2.x-p.x)*(p2.y-p.y);*/

            //Mat_<float> P(2,1);
            //P << p.x, p.y;
            //float P[2][1]={p.x, p.y};

            //Mat_<float> Pc(2,1);
            //Pc<< p.x-newLoc.x, p.y-newLoc.y;
            float Pc[2][1] = {p.x-newLoc.x, p.y-newLoc.y};

            den_buf1[0][0]=1.0-(V_buf_t[0][0]/V_buf);
            den_buf1[0][1]=0.0-(V_buf_t[0][1]/V_buf);
            den_buf1[1][0]=0.0-(V_buf_t[1][0]/V_buf);
            den_buf1[1][1]=1.0-(V_buf_t[1][1]/V_buf);
            //std::cout<< "den_buf1"<< std::endl<< den_buf1[0][0]<<" "<<den_buf1[0][1]<<endl<<den_buf1[1][0]<<" "<<den_buf1[1][1]<<std::endl;
            //subtract(I,(V_buf_t)*(1/V_buf), den_buf1);
            perpendicular[0][0]=den_buf1[0][0]*Pc[0][0]+den_buf1[0][1]*Pc[1][0];
            perpendicular[1][0]=den_buf1[1][0]*Pc[0][0]+den_buf1[1][1]*Pc[1][0];
            //std::cout<< "perpendicular"<< std::endl<< perpendicular[0][0]<<endl<<perpendicular[1][0]<<std::endl;
            //perpendicular = (den_buf1*Pc);

            //float per_dist = std::sqrt(std::pow(perpendicular.at<float>(0,0),2)+std::pow(perpendicular.at<float>(1,0),2));
            float per_dist = std::sqrt(std::pow(perpendicular[0][0],2)+std::pow(perpendicular[1][0],2));

            // Slope equation (y1 - y2) / (x1 - x2)
            /*if (p.x != p2.x)
            {
                float m = (p.y - p2.y) / (p.x - p2.x);
                // Line equation:  y = mx + b
                float b = p.y - (m * p.x);
                per_dist = std::abs((newLoc.y-m*newLoc.x-b)/(std::sqrt(1+std::pow(m,2))));
            }
            else
            {
                per_dist = std::abs(newLoc.x-p.x);
            }*/
            float distance = std::sqrt(std::pow(Pc[0][0],2)+std::pow(Pc[1][0],2));


            //if the line passes very close to the centre, then compute least square coordinate


            if(  (per_dist<3.0)&&(distance<100))
            {

                /*Mat V = (Mat_<float>(2,1) << (p2.x-p.x), (p2.y-p.y));
                float V_buf = V.at<float>(0,0)*V.at<float>(0,0)+V.at<float>(1,0)*V.at<float>(1,0);
                mulTransposed(V, V_buf_t, false);

                Mat_<float> P(2,1);
                P << p.x, p.y;
                subtract(I,(V_buf_t)*(1/V_buf), den_buf1);*/

                //num_buf1 = den_buf1*P;

                num_buf1[0][0]=den_buf1[0][0]*P[0][0]+den_buf1[0][1]*P[1][0];
                num_buf1[1][0]=den_buf1[1][0]*P[0][0]+den_buf1[1][1]*P[1][0];
                num_buf2[0][0] = num_buf2[0][0]+num_buf1[0][0];
                num_buf2[1][0] = num_buf2[1][0]+num_buf1[1][0];

                den_buf2[0][0]=den_buf2[0][0]+den_buf1[0][0];
                den_buf2[0][1]=den_buf2[0][1]+den_buf1[0][1];
                den_buf2[1][0]=den_buf2[1][0]+den_buf1[1][0];
                den_buf2[1][1]=den_buf2[1][1]+den_buf1[1][1];
                   // std::cout<< "num_buf1"<< std::endl<< num_buf1[0][0]<<endl<<num_buf1[1][0]<<std::endl;
                //num_buf = num_buf1+num_buf;
                //den_buf = den_buf1+den_buf;


                //arrowedLine(*srcROI,p2,p,Scalar(0,255,0),0.1,8,0,0.1);

            }

        }

        pixel_x = pixel_x+resolution;
        pixel_y = pixel_y+resolution;
        pixel = pixel+resolution;
         }
    }

   num_buf << num_buf2[0][0], num_buf2[1][0];

   float det_den = den_buf2[0][0]*den_buf2[1][1]-den_buf2[0][1]*den_buf2[1][0];
   //taking inverse of 2x2 matrix
 /*  den_buf2[0][0] = den_buf2[1][1]/det_den;
   den_buf2[0][1] = -den_buf2[0][1]/det_den;
   den_buf2[1][0] = -den_buf2[1][0]/det_den;
   den_buf2[1][1] = den_buf2[0][0]/det_den;*/
    //std::cout<< "num_buf1"<< std::endl<< den_buf1[0][0]<<endl<<den_buf1[1][0]<<std::endl;

   //final_product[0][0]=den_buf2[0][0]*num_buf2[0][0]+den_buf2[0][1]*num_buf2[1][0];
   //final_product[1][0]=den_buf2[1][0]*num_buf2[0][0]+den_buf2[1][1]*num_buf2[1][0];

   den_buf << den_buf2[1][1]/det_den, -den_buf2[0][1]/det_den, -den_buf2[1][0]/det_den, den_buf2[0][0]/det_den;

   //den_buf = den_buf.inv();

   final_product = den_buf*num_buf;
   *X = final_product.at<float>(0,0);
   *Y = final_product.at<float>(1,0);

}
