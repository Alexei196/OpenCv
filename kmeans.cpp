#include <iostream>
#include<vector>
#include <opencv2/opencv.hpp>

using namespace cv;

int distance(const int&, const int&);

int main(int argc, char** argv ) {
    if ( argc != 2 ) {
        printf("usage: kmeans.exe <Image_Path>\n");
        return -1;
    }

    //load image into matrix
    Mat image = imread( argv[1], IMREAD_GRAYSCALE);
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
    //set up for kmeans
    const int clustersCount = 4, iterations = 7;

    //1. Define random centroids for k clusters
    long long int centroids[clustersCount];
    long long int centroidSum[clustersCount];
    int centroidCount[clustersCount];
    for(int i = 0; i < clustersCount; ++i) {
        centroids[i] = (long long int) (rand() % 256);
        centroidSum[i] = 0ll;
        centroidCount[i] = 0;
    }
    //2. Assign data to closest centroid
    int lowestDistance, closestCentroid;
    //For each iteration of the k-means alg
    for(int i = 0; i < iterations; ++i) {
        printf("DEBUG ITER: %d\n", i);
        //For each pixel in image
        for(long int y = 0; y < image.rows; ++y) {
            for(long int x = 0; x < image.cols; ++x) {
                if((int)image.at<unsigned char>(y,x) < 24) {
                    continue;
                }
                //For each centroid in existence
                closestCentroid = 0;
                lowestDistance = distance((int)image.at<unsigned char>(y,x), centroids[0]);
                for(int c = 1; c < clustersCount; ++c) {
                    int space = distance((int)image.at<unsigned char>(y,x), centroids[c]);
                    if(space < lowestDistance) {
                        closestCentroid = c;
                        lowestDistance = space;
                    }
                    
                }
                centroidSum[closestCentroid] += (long long int)image.at<unsigned char>(y,x);
                centroidCount[closestCentroid] += 1;
            }
        }
        //3. Assign centroid to average of each grouped data
        for(int c = 0; c < clustersCount; ++c) {
            centroids[c] = (long long int) (centroidSum[c] / (long long int) centroidCount[c]);
            printf("Centroid %d: %lld\n", c, centroids[c]); 
        }
    }
    //4. perform 2 and 3 i amount of times   

    Mat centroidAssigned = image.clone();

    for(long int y = 0; y < image.rows; ++y) {
        for(long int x = 0; x < image.cols; ++x) {  
            closestCentroid = 0;
            lowestDistance = distance((int)image.at<unsigned char>(y,x), centroids[0]);
            for(int c = 1; c < clustersCount; ++c) {
                int space = distance((int)image.at<unsigned char>(y,x), centroids[c]);
                if(space < lowestDistance) {
                    closestCentroid = c;
                    lowestDistance = space;
                }
            }
            centroidAssigned.at<unsigned char>(y,x) = closestCentroid * 122;
        }
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", centroidAssigned);

    waitKey(0);
    return 0;
}

int distance(const int &l1, const int &l2) {
    return (l2 - l1) < 0 ? -1*(l2-l1) : (l2-l1);
}