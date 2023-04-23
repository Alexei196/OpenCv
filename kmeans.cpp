#include<iostream>
#include<vector>
#include<omp.h>
#include<opencv2/opencv.hpp>

using namespace cv;

int distance(const int&, const int&);
Mat kMeans(const Mat&, const int&, const int&, int);

Mat sobel(const Mat&, int);

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
    //set up for kmeans

    const int portions = 4; 
    Mat imageSlices[portions];

    for(int i = 0; i < portions; ++i) {
        imageSlices[i] = Mat(image, Range((image.rows * i / portions), (image.rows * (i+1) / portions) - 1), Range(0, image.cols));
    }
    printf("sliced\n");
    
    
    //for each chunk perform k-means
    //TODO kmeans must be split up for processes to share centroids
    Mat kMeansImg, sobelImg, newImage;
    for(int my_rank = 0; my_rank < portions; ++my_rank){
        
    }
    kMeansImg = image;
    //kMeansImg = kMeans(imageSlices[0], 3, 3, 4);
    kMeansImg = sobel(kMeansImg, 50);

    //sobelImg = imread("brainRegions.jpg", IMREAD_GRAYSCALE);
    
    printf("sobel \n");

    // for(int my_rank = 1; my_rank < portions; ++my_rank){
    //     vconcat(imageSlices[0], imageSlices[my_rank], imageSlices[0]);
    // }


    namedWindow("Image", WINDOW_AUTOSIZE );
    imshow("Image", kMeansImg);
    waitKey(0);
    return 0;
}

int distance(const int &l1, const int &l2) {
    return (l2 - l1) < 0 ? -1*(l2-l1) : (l2-l1);
}

Mat kMeans(const Mat& image, const int& clustersCount, const int& iterations, int threadCount) {
    //1. Define random centroids for k clusters
    long long int centroidSum[clustersCount];
    int centroids[clustersCount], centroidCount[clustersCount];
    for(int i = 0; i < clustersCount; ++i) {
        centroids[i] = (int) (rand() % 256);
        centroidSum[i] = 0ll;
        centroidCount[i] = 0;
    }

    //2. Assign data to closest centroid
    Mat centroidAssigned = image.clone();
    int lowestDistance, closestCentroid, c;
    int colorScale = 255 / (clustersCount - 1);
    long int y, x;
    //For each iteration of the k-means alg
    for(int i = 0; i < iterations; ++i) {
        //For each pixel in image
        #pragma omp parallel for num_threads(threadCount) \
        default(none) shared(image, centroidAssigned, colorScale, centroids, centroidSum, centroidCount, clustersCount) private(y,x, c, closestCentroid, lowestDistance) 
        for(y = 0; y < image.rows; ++y) {
            for(x = 0; x < image.cols; ++x) {
                // option for centroids to ignore all low value/black pixels
                if((int)image.at<unsigned char>(y,x) < 24) {
                    //continue;
                }
                //For each centroid in existence
                closestCentroid = 0;
                lowestDistance = distance((int)image.at<unsigned char>(y,x), centroids[0]);
                for(c = 1; c < clustersCount; ++c) {
                    int space = distance((int)image.at<unsigned char>(y,x), centroids[c]);
                    if(space < lowestDistance) {
                        closestCentroid = c;
                        lowestDistance = space;
                    }
                }
                //Now that centroids are found, replace the pixels with the color of the centroid
                centroidAssigned.at<unsigned char>(y,x) = closestCentroid * colorScale;
                centroidSum[closestCentroid] += (long long int)image.at<unsigned char>(y,x);
                centroidCount[closestCentroid] += 1;
            }
        }
        //3. Assign centroid to the average of each grouped data
        for(int c = 0; c < clustersCount; ++c) {
            if(centroidCount[c] == 0) {
                //In event centroid is not counted
                fprintf(stderr, "Centroid %d did not gain any points!\n", c);
                continue;
            }
            centroids[c] = (long long int) (centroidSum[c] / (long long int) centroidCount[c]);
        }
    }
    //4. perform 2 and 3 i amount of times   
    return centroidAssigned;
}

//TODO sobel experiences errors when used to set allocated Mat

Mat sobel(const Mat& gray_img, int threshold) {
    Mat G, x, y;
    Mat sobel_img = gray_img.clone();
    if(threshold < 256) {
        threshold*= threshold;
    }

    for (int row = 0; row < sobel_img.rows; row++)
    {
        for (int col = 0; col < 512; col++)
        {
            printf("DEBUG  ");
            if (row >= sobel_img.rows - 2 || col >= sobel_img.cols - 2){
                printf("IF(%d,%d), ",col, row);
                sobel_img.at<unsigned char>(col,row) = 0;
                continue;
            }
            // Mat G;
            // gray_img(cv::Rect(row, col, 3, 3)).copyTo(G);
            // G.convertTo(G, CV_32SC1);
            G = (Mat_<int>(3, 3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);
            x = (Mat_<int>(3, 3) << 1, 0, -1, 2, 0, -2, 1, 0, -1);
            y = (Mat_<int>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
            
            double G_x = sum(G.mul(x))[0];
            printf("SUMX:%f, ", G_x);
            double G_y = sum(G.mul(y))[0];
            printf("SUMY:%f, ", G_y);
            double pixel = pow(G_x, 2) + pow(G_y, 2);
            if (pixel <= threshold)
                pixel = 0;
            else 
                pixel = 255;
            printf(" SU(%d,%d), SIZE: %d",col, row, sobel_img.cols);
            sobel_img.at<unsigned char>(col,row) = pixel;
            printf(" END_DEBUG\n");
        }
    }
    return sobel_img;
    
}