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
    Mat image;
    image = imread( argv[1], 1 );
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

    bool loop = true;
    int i, k;
    int sampleCount, clusterCount = 3;
    Mat points(sampleCount, 1, CV_32FC2), labels;
    std::vector<Point2f> centers;
    //kmeans function call from OpenCV 
    double compactness = kmeans(image, clusterCount, labels, 
    TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0), 
    3, KMEANS_PP_CENTERS, centers);

    for(i = 0; i < sampleCount; ++i) {
        int clusterIdx = labels.at<int>(i);
        Point ipt = points.at<Point2f>(i);
        circle(image, ipt, 2, colorTab[clusterIdx], FILLED, LINE_AA);
    }

    for(i = 0; i < (int) centers.size(); ++i) {
        Point2f c = centers[i];
        circle(image, c, 40, colorTab[i], 1, LINE_AA);
    }

    //display results
    std::cout << "Compactness: " << compactness << "\n"; 
    imshow("clusters", image);

    
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);

    
    waitKey(0);
    return 0;
}