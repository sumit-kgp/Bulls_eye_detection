#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <time.h>

//User defined functions/constructors

#include "calcmaxintensity.h"
#include "addstraightline.h"
#include "least_square_detection.h"
#include "least_square_detection2.h"
#include "multiple_tracking.h"
#include "suppress_local_neighborhood.h"
#include "non_maxima_suppression.h"
#include "non_maxima_suppression2.h"

using namespace cv;
using namespace std;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    clock_t start, end;
        double elapsed;


      Mat src, src_float;
      Mat grad;

      int scale = 1;
      int delta = 0;
      int ddepth = CV_32F;


    for(int ij=0;ij<10;ij++)                         //SPEED TEST FOR 10 ITERATIONS
     {
     src = imread("/home/sumit/Downloads/basler2.jpg");
     start = clock();
     Mat src_gray;
     cv::resize(src, src_gray, Size(), 0.25, 0.25, INTER_NEAREST);

      /// Convert it to gray
      cvtColor( src_gray, src_gray, CV_BGR2GRAY );
      //GaussianBlur( src_gray, src_gray, Size(15,15), 5, 5, BORDER_DEFAULT );
      //GaussianBlur( src_gray, src_gray, Size(5,5), 2, 2, BORDER_DEFAULT );//when size is petite
      GaussianBlur( src_gray, src_gray, Size(5,5), 4, 4, BORDER_DEFAULT );

      src_gray.convertTo(src_float,CV_32FC1);
      /// Generate grad_x and grad_y
      Mat grad_x, grad_y;

      /// Gradient X
      Scharr( src_float, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );

      /// Gradient Y
      Scharr( src_float, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );

      /// Total Gradient (approximate)

        Mat mask = Mat::zeros(src_float.size(), CV_32FC1);
        grad = Mat::zeros(src_float.size(), CV_32FC1);

        int resolution = 1;

        calcMaxIntensity(src_gray, grad_x, grad_y, &grad);                  //Computes overall gradient magnitude

        double minVal, maxVal;
        //minMaxLoc(grad, &minVal, &maxVal, NULL, NULL);

        for (int i = 0 ; i < src_gray.rows ; i+= resolution)
        {

            float* pixel_x = grad_x.ptr<float>(i);
            float* pixel_y = grad_y.ptr<float>(i);
            float* pixel = grad.ptr<float>(i);

                    for (int j = 0 ; j < src_gray.cols ; j+= resolution)
                    {
                    Point2f p(j,i),p2;
                    Point2f grad_ptr;
                    grad_ptr.x = *pixel_x;
                    grad_ptr.y = *pixel_y;

                            if(*pixel>250){  //std::cout<<"maxLoc is"<<maxLoc<<std::endl; //350 for 2048 image

                            p2.x = p.x+grad_ptr.x;                                                      //Creating vectors of length gradient intensity
                            p2.y = p.y+grad_ptr.y;

                                 addStraightLine(&mask, p2, p, *pixel);     //Function to add straight lines to each pixel weight to its respective intensity level
                            }
                    pixel_x = pixel_x+resolution;
                    pixel_y = pixel_y+resolution;
                    pixel = pixel+resolution;
                    }
        }


        minMaxLoc(mask, &minVal, &maxVal, NULL, NULL);

            //Scaling down the image values

        mask.convertTo(mask,CV_8U, 255/(maxVal-minVal),-minVal*255/(maxVal-minVal));


        Mat mask2;
        suppress_local_neighborhood(mask, &mask2);
        imshow("final bitwise", mask2);

           end = clock();
           elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
           fprintf(stdout,"time elapsed before multiple tracking is %f\n", elapsed);

        //MULTIPLE TRACKING
        std::vector<Point> p_temp;
        //multiple_tracking(mask2, dst, maxVal, &p_temp);


        //NON MAXIMA SUPPRESSION
        //for(int i = 0; i<2; i++)
        //non_maxima_suppression(&mask2, &p_temp);
        non_maxima_suppression2(mask2, &p_temp);

        end = clock();
        elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
        fprintf(stdout,"time elapsed after multiple tracking is %f\n", elapsed);


      /*  for (int n = 0; n < p_temp.size(); n++)
        {
            std::cout << "points of local maxima" << std::endl << p_temp[n].x << "," <<p_temp[n].y<< std::endl;
        }*/

        //  END OF MULTIPLE TRACKING




            Point maxLoc;

            float xscale = 4.0;//src.cols/800;
            float yscale = 4.0;//src.rows/800;

            imshow("Intersecting lines", mask);


            for (int n = 0; n < 20; n++)
                {

                maxLoc.x = xscale*p_temp[n].x;
                maxLoc.y = yscale*p_temp[n].y;

                //circle(src, maxLoc, 4, Scalar(255,0,0), 4, 8, 0);             //DISPLAY THE PRELIMINARY CENTER

                fprintf(stdout,"Maximum intensity is %f @ %d,%d\n",maxVal, maxLoc.x, maxLoc.y);

                //script for printing lines closer to the centre

                Point newLoc;
                float offset = 100.0;
                newLoc.x=offset;
                newLoc.y=offset;

                    if((maxLoc.y>=offset)&&(maxLoc.y+offset<=src.rows)&&(maxLoc.x>=offset)&&(maxLoc.x+offset<=src.cols))
                        {

                            Mat srcROI(src,cv::Rect(maxLoc.x-offset,maxLoc.y-offset,2*offset,2*offset));


                 //GaussianBlur( srcROI, srcROI, Size(5,5), 2, 2, BORDER_DEFAULT );
                 /// Gradient X
                 Scharr( srcROI, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );

                 /// Gradient Y
                 Scharr( srcROI, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );

                 grad = Mat::zeros(srcROI.size(), CV_32FC1);

                float X, Y;

                //printing lines close to centre
                least_square_detection2(&srcROI, grad_x, grad_y, grad, newLoc, &X, &Y);                  //does least square computation for nearly concentric vectors

                end = clock();
                elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
                fprintf(stdout,"time elapsed after every least square operation %f\n", elapsed);

                float X_src=(maxLoc.x-offset+X);
                float Y_src=(maxLoc.y-offset+Y);
                circle(src, Point(X_src,Y_src), 3, Scalar(0,0,255), 3, 8, 0);
                fprintf(stdout,"location is %f, %f\n", X_src, Y_src);
                        }
            }
        imshow("equalizedonbig", src);

        end = clock();
        elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
        fprintf(stdout,"time elapsed is %f\n\n", elapsed);


      } //FOR 10 ITERATIONS

}//END OF MAIN

MainWindow::~MainWindow()
{
    delete ui;
}
