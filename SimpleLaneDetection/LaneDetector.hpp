//
//  LaneDetector.hpp
//  SimpleLaneDetection
//
//  Created by Anurag Ajwani on 28/04/2019.
//  Copyright © 2019 Anurag Ajwani. All rights reserved.
//

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class LaneDetector {
    
    public:
    
    //Constructor
    LaneDetector(){
        srcVertices[0] = cv::Point(700, 605);
        srcVertices[1] = cv::Point(890, 605);
        srcVertices[2] = cv::Point(1760, 1030);
        srcVertices[3] = cv::Point(20, 1030);

        dstVertices[0] = cv::Point(0, 0);
        dstVertices[1] = cv::Point(640, 0);
        dstVertices[2] = cv::Point(640, 480);
        dstVertices[3] = cv::Point(0, 480);
    }
    
    /*
     Returns image with lane overlay
     */
    Mat detect_lane(Mat image);
    
    private:
    Point2f srcVertices[4];
    Point2f dstVertices[4];
    /*
     Filters yellow and white colors on image
     */
    Mat filter_only_yellow_white(Mat img, Mat dst);
    
    /*
     Crops region where lane is most likely to be.
     Maintains image original size with the rest of the image blackened out.
     */
    Mat crop_region_of_interest(Mat image);
    
    /*
     Draws road lane on top image
     */
    Mat draw_lines(Mat image, vector<Vec4i> lines);
    
    /*
     Detects road lanes edges
     */
    Mat detect_edges(Mat image);
    
    tuple<Mat, Mat> transformProspectives(Mat image);
    
    vector<Point2f> slidingWindow(Mat image, cv::Rect window);

};
