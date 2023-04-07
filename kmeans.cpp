#include <iostream>
#include<vector>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char** argv ) {
    if ( argc != 2 ) {
        printf("usage: kmeans.exe <Image_Path>\n");
        return -1;
    }

    //load image into matrix
    Mat image = imread( argv[1], 1 );
    if ( !image.data ) {
        printf("No image data \n");
        return -1;
    }
    //Setup for k means analysis
    Scalar colorTab[] = {
        Scalar(255, 0, 0),
        Scalar(0, 255, 0),
        Scalar(0, 0, 255)
    };
    //
    int clustersCount = 3, iterations = 5;
    Mat segmentsImage = imageQuantization(image, clustersCount, iterations);

    
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);

    
    waitKey(0);
    return 0;
}