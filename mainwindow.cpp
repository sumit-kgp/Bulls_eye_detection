#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <time.h>
#include <unistd.h>
#include "opencv2/videoio.hpp"
#include <boost/filesystem.hpp>
#include <eigen3/Eigen/Dense>


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
using namespace boost::filesystem;
using namespace Eigen;

ofstream monotrackingFile;

///FUNCTION TO READ IMAGES FROM A DESIGNATED FOLDER
void readFilenamesBoost(vector<string> &filenames, const string &folder)
{
    path directory(folder);
    directory_iterator itr(directory), end_itr;

    string current_file = itr->path().string();

    for(;itr != end_itr; ++itr)
    {
// If it's not a directory, list it. If you want to list directories too, just remove this check.
       if (is_regular_file(itr->path()))
       {
       // assign current file name to current_file and echo it out to the console.
//            string full_file_name = itr->path().string(); // returns full path
            string filename = itr->path().filename().string(); // returns just filename
            filenames.push_back(filename);
       }
    }
}

///FUNCTION TO CALCULATE EULER ROTATIONAL (3,3) MATRIX FROM ANGLE AXIS (3) REPRESENTATION
void calcRodrigues(Matrix<float, 3, 1> W, Matrix<float, 3, 3>* R){

    Matrix<float, 1, 3> W_t = W.transpose();

    float Omega_norm = sqrt(W_t*W);//sqrt(W[0]*W[0]+W[1]*W[1]+W[2]*W[2]);

    Matrix<float, 3, 3> Skew_Omega;
    Skew_Omega << 0, -W[2], W[1],
                  W[2], 0, -W[0],
                  -W[1], W[0], 0;

    Matrix<float, 3, 3> Skew_Omega2 = W*W_t-pow(Omega_norm,2)*Matrix3f::Identity(3,3);/*{W[0]*W[0]-pow(Omega_norm,2),        W[0]*W[1],      W[0]*W[2],
                                W[1]*W[0],      W[1]*W[1]-pow(Omega_norm,2),        W[1]*W[2],
                                W[2]*W[0],      W[2]*W[1],      W[2]*W[2]-pow(Omega_norm,2)};*/

    float sine = sin(Omega_norm)/Omega_norm;
    float cosine = (1-cos(Omega_norm))/pow(Omega_norm,2);

    *R = Matrix3f::Identity(3,3) + sine*Skew_Omega+cosine*Skew_Omega2;

    /*double rotation[3][3] = {1+sine*Skew_Omega[0][0]+cosine*Skew_Omega2[0][0],    sine*Skew_Omega[0][1]+cosine*Skew_Omega2[0][1],     sine*Skew_Omega[0][2]+cosine*Skew_Omega2[0][2],
               sine*Skew_Omega[1][0]+cosine*Skew_Omega2[1][0],      1+sine*Skew_Omega[1][1]+cosine*Skew_Omega2[1][1],   sine*Skew_Omega[1][2]+cosine*Skew_Omega2[1][2],
               sine*Skew_Omega[2][0]+cosine*Skew_Omega2[2][0],      sine*Skew_Omega[2][1]+cosine*Skew_Omega2[2][1],     1+sine*Skew_Omega[2][2]+cosine*Skew_Omega2[2][2]};*/

}

///FUNCTION TO TRAINGULATE SPATIAL COORDINATES USING CAMERA TRANSFORMATION AND PIXEL VALUES
void triangulate(Point UV , Matrix<float, 2, 3> S, Matrix<float, 3, 3> R, Matrix<float, 3, 1> T, Matrix<float, 3, 1>* X){

    Matrix<float, 2, 1> uv(UV.x, UV.y);
    Matrix<float, 2, 3> temp = S*R;
    Matrix<float, 3, 2> MoorePenrose = (temp.transpose())*(temp*temp.transpose()).inverse();
    *X = MoorePenrose*(uv-S*T);
}


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    clock_t start, end;
    double elapsed;


    Mat src_float, templBackground, tempBackground, background;
    Mat grad;
    Point Lock;


//    ofstream monotrackingFile;

    int scale = 1, background_buffer_counter = 0, num_buff_background =10;
    int delta = 0;
    int ddepth = CV_32F;

    ///INITIALIZING CAMERA TRANSFORMATION MATRIX
    ///
    Matrix<float,3,3>R1;
    Matrix<float,3,3>R2;

    Matrix<float, 3, 1> W1(3,1);
    W1 << -1.1991, -1.2237, 1.215;

    Matrix<float, 3, 1> W2(3,1);
    W2 << 0.0127, 0.0097, 1.5813;

    Matrix<float, 3, 1> T1(3,1);
    T1 << 4.6392, 2.9923, 0;

    Matrix<float, 3, 1> T2(3,1);
    T2 << 5.6624, 5.5949, 0;

    Matrix<float, 2, 3> S1;
    S1<< 208.3,0,0,
         0,208.3,0;

    Matrix<float, 2, 3> S2;
    S2<< 181.8,0,0,
         0,181.8,0;

    calcRodrigues(W2, &R2);
    calcRodrigues(W1, &R1);

    Matrix<float,3,1> X;
    ///BACKGROUND SUBTRACTION FOR TEST IMAGE

    //src = imread("/home/sumit/Downloads/basler2.jpg");
    /*src = imread("/media/sumit/44AA78DBAA78CAC6/ETH Zürich/Nanoscribe/20Apr2016_holograms/6_closeup.bmp");
    //std::cout<< "raw"<< std::endl<< src<<std::endl;
    //imshow("original image", src);
    //cvtColor( src, src, CV_BGR2GRAY );
    src.convertTo(src,CV_32FC1);
    //std::cout<< "raw_convert32"<< std::endl<< src<<std::endl;

    src2 = imread("/media/sumit/44AA78DBAA78CAC6/ETH Zürich/Nanoscribe/20Apr2016_holograms/backafter5.bmp");
    src2.convertTo(src2,CV_32FC1);
    //std::cout<< "back_convert32"<< std::endl<< src2<<std::endl;

    //cvtColor( src2, src2, CV_BGR2GRAY );
    cv::subtract(src,src2,src);
    //std::cout<< "subtract"<< std::endl<< src<<std::endl;
    cv::normalize(src, src, 0, 255, NORM_MINMAX);
    //std::cout<< "subtract_normalize"<< std::endl<< src<<std::endl;
    src.convertTo(src, CV_8U);

    //imshow("subtracted image", src);
    cv::Rect myROI(200, 100, 1500, 1100);
    src = src(myROI);
    imshow("subtracted image", src);*/



    ///CAPTURING FROM A DIRECTORY
    string folder = "/media/sumit/UUI/multiple_tracking/cam2/";
    char file[256];
    vector<string> filenames;

    readFilenamesBoost(filenames, folder);    
    std::sort(filenames.begin(), filenames.end());
    monotrackingFile.open("/home/sumit/TrackerImages/monotracking.xls");
    monotrackingFile<<"filename" <<"\t"<< "U" <<"\t" << "V" << "\n";

    for(size_t i = 0; i < filenames.size(); ++i)
    {
           Mat src = imread(folder + filenames[i]);
            cvtColor( src, src, CV_BGR2GRAY );
       //      if (waitKey(0) == 27) break;
           if(!src.data)
               cerr << "Problem loading image!!!" << endl;
           background_buffer_counter++;
//}


    ///CAPTURING FROM A VIDEO FILE
//    VideoCapture cap("/media/sumit/44AA78DBAA78CAC6/ETH Zürich/Pylon/12Aug2016calibcam2.avi");//21Jul2016_abf_1_cam_2_raw.mp4.avi");
//    //VideoCapture cap(0);
//    if ( !cap.isOpened() )  // if not success, exit program
//    {
//        cout << "Cannot open the video file" << endl;
//        //return -1;
//    }

//    double count = cap.get(CV_CAP_PROP_FRAME_COUNT); //get the frame count
//    cap.set(CV_CAP_PROP_POS_FRAMES,0); //Set index to last frame
//    namedWindow("MyVideo",CV_WINDOW_AUTOSIZE);

    ///if background subtraction isn't required
      // num_buff_background = -1;
      // tempBackground = Mat::zeros(src.rows, src.cols, CV_32FC1);

    ///BACKGROUND SUBTRACTION


       //cvtColor( src, background, CV_BGR2GRAY );


    if (background_buffer_counter <= num_buff_background){

         src.convertTo(background,CV_32FC1);
//        cap.read(background);
//        cvtColor( src, background, CV_BGR2GRAY );
         if (background_buffer_counter == 1)
        tempBackground = Mat::zeros(background.rows, background.cols, CV_32FC1);


        ///BUFFER BACKGROUND SUBTRACTION
        std::vector<cv::Mat> bufferBackground;

        //for (int i=0;i<num_buff_background;i++)
        //{
            //Mat background;
            //cap.read(background);
            //cvtColor( background, background, CV_BGR2GRAY );

            //background.convertTo(background,CV_32FC1);
            //medianBlur(background, background, 3 );

            bufferBackground.push_back(1/(float)num_buff_background*background);
            tempBackground += 1/(float)num_buff_background*background;

        //}
        //break;
    }

    else if(background_buffer_counter > num_buff_background){

start = clock();
    //cap.set(CV_CAP_PROP_POS_FRAMES,num_buff_background); //Set index to first frame or num_buff_background frame if above chunk of code is active

    //while(1)
    //{

        double alpha = 0.9;
        Mat frame;

        //bool success = cap.read(frame);
        //cvtColor( src, frame, CV_BGR2GRAY );
//        if (!success){
//            cout << "Cannot read  frame " << endl;
//            break;
//        }

         src.convertTo(frame,CV_32FC1);
        //start = clock();
        ///IMPLEMENTING REAL TIME MOVING AVG BACKGROUND SUBTRACTION

       // if (cap.get(CV_CAP_PROP_POS_FRAMES)!= num_buff_background+1)
        //{
         if(background_buffer_counter > num_buff_background+1)
           tempBackground = alpha * tempBackground + (1-alpha) * templBackground;

        //}



        ///COPYING THE FRAME FOR SUBSEQUENT BACKGROUND SUBTRACTION
         frame.copyTo(templBackground);

        cv::subtract(frame, tempBackground,frame);
        cv::normalize(frame, src, 0, 255, NORM_MINMAX);

        end = clock();
        elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
        fprintf(stdout,"time elapsed in background subtraction %f\n", elapsed);

        waitKey(1);
        //if (waitKey(0) == 27) break;
        //}

        end = clock();
        elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
        fprintf(stdout,"time elapsed after waitkey %f\n", elapsed);

        Mat src_gray;
        //GaussianBlur( src, src, Size(5,5), 3, 3, BORDER_DEFAULT );
        cv::resize(src, src_gray, Size(), 0.25, 0.25, INTER_NEAREST);

        /// Convert it to gray
        //cvtColor( src_gray, src_gray, CV_BGR2GRAY );
       // GaussianBlur( src_gray, src_gray, Size(21,21), 7, 7, BORDER_DEFAULT );
        //GaussianBlur( src_gray, src_gray, Size(5,5), 2, 2, BORDER_DEFAULT );//when size is petite
        //
       GaussianBlur( src_gray, src_gray, Size(5,5), 4, 4, BORDER_DEFAULT );
         //GaussianBlur( src_gray, src_gray, Size(5,5), 5, 5, BORDER_DEFAULT );
         //GaussianBlur( src_gray, src_gray, Size(7,7), 6, 6, BORDER_DEFAULT );

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

        int resolution =1;
        double minVal, maxVal;
        calcMaxIntensity(src_gray, grad_x, grad_y, &grad);                  //Computes overall gradient magnitude
        imshow("gradient", grad);
        /*Mat back ;
        back = grad.clone(); //Mat::zeros(grad.size(), grad.type());
        //double minVal, maxVal;
        Mat gdm = back.clone();
        minMaxLoc(back, &minVal, &maxVal, NULL, NULL);

            //Scaling down the image values

        back.convertTo(back,CV_8U, 255/(maxVal-minVal),-minVal*255/(maxVal-minVal));
        imshow("after background filtering", back);
        imwrite("/home/sumit/Desktop/tracking image/gradient_based_segmentation.jpg", back);*/


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

                if(*pixel>200 ){  //std::cout<<"maxLoc is"<<maxLoc<<std::endl; //350 for 2048 image   //250 for 1252 image //(*pixel>0 && *pixel<400) camcalib2

                    p2.x = p.x+grad_ptr.x;                                                      //Creating vectors of length gradient intensity
                    p2.y = p.y+grad_ptr.y;

                    ///Function to add straight lines to each pixel weight to its respective intensity level
                    addStraightLine(&mask, p2, p, *pixel);
                    //arrowedLine(gdm,p2,p,Scalar(255,255,255),0.1,8,0,0.1);

                }
                pixel_x = pixel_x+resolution;
                pixel_y = pixel_y+resolution;
                pixel = pixel+resolution;
            }
        }

        //imwrite("/home/sumit/Desktop/tracking image/gradient_direction_matching.jpg", gdm);
        GaussianBlur( mask, mask, Size(25,25), 5, 5, BORDER_DEFAULT );//when size is petite


        minMaxLoc(mask, &minVal, &maxVal, NULL, NULL);

        ///Scaling down the image values

        mask.convertTo(mask,CV_8U, 255/(maxVal-minVal),-minVal*255/(maxVal-minVal));


        Mat mask2;
        suppress_local_neighborhood(mask, &mask2, background_buffer_counter);
        imshow("final bitwise", mask2);
        sprintf(file,"/home/sumit/Desktop/PPT_Videos/SLN/sln_%d.png\n", background_buffer_counter);
        //imwrite(file, mask2);
        //imwrite("/home/sumit/Desktop/tracking image/after_suppress_local.jpg", mask2);

        end = clock();
        elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
        fprintf(stdout,"time elapsed before multiple tracking is %f\n", elapsed);

        ///MULTIPLE TRACKING
        std::vector<Point> p_temp;
        //multiple_tracking(mask2, dst, maxVal, &p_temp);


        ///NON MAXIMA SUPPRESSION
        //for(int i = 0; i<2; i++)
        //non_maxima_suppression(&mask2, &p_temp);
        non_maxima_suppression2(mask2, &p_temp, background_buffer_counter);

        end = clock();
        elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
        fprintf(stdout,"time elapsed after multiple tracking is %f\n", elapsed);


        /*  for (int n = 0; n < p_temp.size(); n++)
        {
            std::cout << "points of local maxima" << std::endl << p_temp[n].x << "," <<p_temp[n].y<< std::endl;
        }*/

        ///  END OF MULTIPLE TRACKING

        Point maxLoc;

        float xscale = 4.0;//src.cols/800;
        float yscale = 4.0;//src.rows/800;

        imshow("Intersecting lines", mask);
        //imwrite("/home/sumit/Desktop/tracking image/voting.jpg", mask);
         sprintf(file,"/home/sumit/Desktop/PPT_Videos/Voting/voting_%d.png\n", background_buffer_counter);
         //imwrite(file, mask);

        ///SELECTING NUMBER OF POINTS FOR TRACKING


        for (int n = 0; n < 5; n++)
        {


            maxLoc.x = xscale*p_temp[n].x;
            maxLoc.y = yscale*p_temp[n].y;

            //circle(src, maxLoc, 4, Scalar(255,0,0), 4, 8, 0);             //DISPLAY THE PRELIMINARY CENTER

            fprintf(stdout,"Maximum intensity is %f @ %d,%d\n",maxVal, maxLoc.x, maxLoc.y);

            //script for printing lines closer to the centre

            Point newLoc;
            float offset = 75.0;
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

                float X1, Y1;

                //printing lines close to centre
                least_square_detection2(&srcROI, grad_x, grad_y, grad, newLoc, &X1, &Y1);                  //does least square computation for nearly concentric vectors

                end = clock();
                elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
                fprintf(stdout,"time elapsed after every least square operation %f\n", elapsed);

                float X_src=(maxLoc.x-offset+X1);
                float Y_src=(maxLoc.y-offset+Y1);

                Lock.x = X_src;    Lock.y = Y_src;

                circle(src, Point(X_src,Y_src), 3, Scalar(0,0,255), 3, 8, 0);
                fprintf(stdout,"location is %f, %f\n", X_src, Y_src);
                //std::cout<<"location is "<<Lock<<endl;

                //monotrackingFile<<i<< "\t"<< X_src <<"\t" << Y_src << "\n";
                triangulate(Lock, S2, R2, T2, &X);

                monotrackingFile<<i<< "\t"<< Lock.x<< "\t"<< Lock.y <<"\t"<< X[0] <<"\t"<< X[1] <<"\t"<< X[2] << "\t\t";

            }
        }
        monotrackingFile<<"\n";

         src.convertTo(src,CV_8U);
        imshow("equalizedonbig", src);
        sprintf(file,"/home/sumit/Desktop/tracking image/cam1pro/raw_%d.png\n", background_buffer_counter);
        imwrite(file, src);
        //imwrite("/home/sumit/Desktop/tracking image/centroid.jpg", src);


        end = clock();
        elapsed = ((double) ( (end - start))) / CLOCKS_PER_SEC;
        fprintf(stdout,"time elapsed is %f\n\n", elapsed);

        //triangulate(Lock, S1, R1, T1, &X);
        //cout<<X<<endl;
        //monotrackingFile<<i<< "\t"<< Lock.x<< "\t"<< Lock.y <<"\t"<< X[0] <<"\t"<< X[1] <<"\t"<< X[2] << "\n";

        end = clock();
        elapsed = ((double) ( (end - start))) / CLOCKS_PER_SEC;
        fprintf(stdout,"time elapsed after triangulation is %f\n\n", elapsed);
        // monotrackingFile.close();

    }/// ELSE IF FOR ACTUAL TRACKING PART
    } ///FOR LOOP FOR READING THE WHOLE DIRECTORY
    monotrackingFile.close();


}///END OF MAIN

MainWindow::~MainWindow()
{
    delete ui;
}
