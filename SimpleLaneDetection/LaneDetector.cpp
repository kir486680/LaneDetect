//
//  LaneDetector.cpp
//  SimpleLaneDetection
//
//  Created by Anurag Ajwani on 28/04/2019.
//  Copyright © 2019 Anurag Ajwani. All rights reserved.
//

#include "LaneDetector.hpp"

using namespace cv;
using namespace std;

double getAverage(vector<double> vector, int nElements) {
    
    double sum = 0;
    int initialIndex = 0;
    int last30Lines = int(vector.size()) - nElements;
    if (last30Lines > 0) {
        initialIndex = last30Lines;
    }
    
    for (int i=(int)initialIndex; i<vector.size(); i++) {
        sum += vector[i];
    }
    
    int size;
    if (vector.size() < nElements) {
        size = (int)vector.size();
    } else {
        size = nElements;
    }
    return (double)sum/size;
}

Mat LaneDetector::detect_lane(Mat image) {
    Mat org;
    Mat perspectiveImg, invertedPerspectiveMatrix = transformProspectives(image);
    Mat processed= filter_only_yellow_white(image, perspectiveImg);
    vector<Point2f> pts = slidingWindow(processed, Rect(0, 420, 120, 60));
    vector<Point> allPts; //Used for the end polygon at the end.
    vector<Point2f> outPts;
    perspectiveTransform(pts, outPts, invertedPerspectiveMatrix); //Transform points back into original image space
    //Draw the points onto the out image
    for (int i = 0; i < outPts.size() - 1; ++i)
    {
        line(org, outPts[i], outPts[i + 1], Scalar(255, 0, 0), 3);
        allPts.push_back(Point(outPts[i].x, outPts[i].y));
    }

    allPts.push_back(Point(outPts[outPts.size() - 1].x, outPts[outPts.size() - 1].y));

    Mat out;
    cvtColor(processed, out, COLOR_GRAY2BGR); //Conver the processing image to color so that we can visualise the lines
    for (int i = 0; i < pts.size() - 1; ++i) //Draw a line on the processed image
    line(out, pts[i], pts[i + 1], Scalar(255, 0, 0));

    //Sliding window for the right side
    pts = slidingWindow(processed, Rect(520, 420, 120, 60));
    perspectiveTransform(pts, outPts, invertedPerspectiveMatrix);

    //Draw the other lane and append points
    for (int i = 0; i < outPts.size() - 1; ++i)
    {
        line(org, outPts[i], outPts[i + 1], Scalar(0, 0, 255), 3);
        allPts.push_back(Point(outPts[outPts.size() - i - 1].x, outPts[outPts.size() - i - 1].y));
    }

    allPts.push_back(Point(outPts[0].x - (outPts.size() - 1) , outPts[0].y));

    for (int i = 0; i < pts.size() - 1; ++i)
        line(out, pts[i], pts[i + 1], Scalar(0, 0, 255));

    //Create a green-ish overlay
    vector<vector<Point>> arr;
    arr.push_back(allPts);
    Mat overlay = Mat::zeros(org.size(), org.type());
    fillPoly(overlay, arr, Scalar(0, 255, 100));
    addWeighted(org, 1, overlay, 0.5, 0, org); //Overlay it

    return org;
}

Mat LaneDetector::filter_only_yellow_white(Mat img, Mat dst) {

    cvtColor(dst, img, COLOR_RGB2GRAY);
    Mat maskYellow, maskWhite;
    inRange(img, Scalar(20, 100, 100), Scalar(30, 255, 255), maskYellow);
    inRange(img, Scalar(150, 150, 150), Scalar(255, 255, 255), maskWhite);

    Mat mask, processed;
    bitwise_or(maskYellow, maskWhite, mask); //Combine the two masks
    bitwise_and(img, mask, processed); //Extract what matches
    
    return processed;
}

Mat LaneDetector::crop_region_of_interest(Mat image) {
    
    /*
     The code below draws the region of interest into a new image of the same dimensions as the original image.
     The region of interest is filled with the color we want to filter for in the image.
     Lastly it combines the two images.
     The result is only the color within the region of interest.
     */
    
    int maxX = image.rows;
    int maxY = image.cols;
    
    Point shape[1][5];
    shape[0][0] = Point(0, maxX);
    shape[0][1] = Point(maxY, maxX);
    shape[0][2] = Point((int)(0.75 * maxY), (int)(0.3 * maxX));
    shape[0][3] = Point((int)(0.25 * maxY), (int)(0.3 * maxX));
    shape[0][4] = Point(0, maxX);
    
    Scalar color_to_filter(255, 255, 255);
    
    Mat filledPolygon = Mat::zeros(image.rows, image.cols, CV_8UC3); // empty image with same dimensions as original
    const Point* polygonPoints[1] = { shape[0] };
    int numberOfPoints[] = { 5 };
    int numberOfPolygons = 1;
    fillPoly(filledPolygon, polygonPoints, numberOfPoints, numberOfPolygons, color_to_filter);
    
    // Cobine images into one
    Mat maskedImage;
    bitwise_and(image, filledPolygon, maskedImage);
    
    return maskedImage;
}

Mat LaneDetector::draw_lines(Mat image, vector<Vec4i> lines) {
    
    vector<double> rightSlope, leftSlope, rightIntercept, leftIntercept;
    
    for (int i=0; i<lines.size(); i++) {
        Vec4i line = lines[i];
        double x1 = line[0];
        double y1 = line[1];
        double x2 = line[2];
        double y2 = line[3];
        
        double yDiff = y1-y2;
        double xDiff = x1-x2;
        double slope = yDiff/xDiff;
        double yIntecept = y2 - (slope*x2);
        
        if ((slope > 0.3) && (x1 > 500)) {
            rightSlope.push_back(slope);
            rightIntercept.push_back(yIntecept);
        } else if ((slope < -0.3) && (x1 < 600)) {
            leftSlope.push_back(slope);
            leftIntercept.push_back(yIntecept);
        }
    }
    
    double leftAvgSlope = getAverage(leftSlope, 30);
    double leftAvgIntercept = getAverage(leftIntercept, 30);
    double rightAvgSlope = getAverage(rightSlope, 30);
    double rightAvgIntercept = getAverage(rightIntercept, 30);
    
    int leftLineX1 = int(((0.65*image.rows) - leftAvgIntercept)/leftAvgSlope);
    int leftLineX2 = int((image.rows - leftAvgIntercept)/leftAvgSlope);
    int rightLineX1 = int(((0.65*image.rows) - rightAvgIntercept)/rightAvgSlope);
    int rightLineX2 = int((image.rows - rightAvgIntercept)/rightAvgSlope);
    
    Point shape[1][4];
    shape[0][0] = Point(leftLineX1, int(0.65*image.rows));
    shape[0][1] = Point(leftLineX2, int(image.rows));
    shape[0][2] = Point(rightLineX2, int(image.rows));
    shape[0][3] = Point(rightLineX1, int(0.65*image.rows));
    
    const Point* polygonPoints[1] = { shape[0] };
    int numberOfPoints[] = { 4 };
    int numberOfPolygons = 1;
    Scalar fillColor(0, 0, 255);
    fillPoly(image, polygonPoints, numberOfPoints, numberOfPolygons, fillColor);
    
    Scalar rightColor(0,255,0);
    Scalar leftColor(255,0,0);
    line(image, shape[0][0], shape[0][1], leftColor, 10);
    line(image, shape[0][3], shape[0][2], rightColor, 10);
    
    return image;
}

Mat LaneDetector::detect_edges(Mat image) {
    
    Mat greyScaledImage;
    cvtColor(image, greyScaledImage, CV_RGB2GRAY);
    
    Mat edgedOnlyImage;
    Canny(greyScaledImage, edgedOnlyImage, 50, 120);
    
    return edgedOnlyImage;
}

Mat LaneDetector::transformProspectives(Mat image){

    Point2f inputQuad[4];
    // Output Quadilateral or World plane coordinates
    Point2f outputQuad[4];
        
    // Lambda Matrix
    //Input and Output Image;
    Mat input, output;
    
    
    //Load the image
    // Set the lambda matrix the same type and size as input
    Mat dst(480, 640, CV_8UC3);
    //For transforming back into original image space Mat invertedPerspectiveMatrix;
    
    // Get the Perspective Transform Matrix i.e. lambda
    
    //Mat perspectiveMatrix = getPerspectiveTransform(inputQuad, outputQuad);
    Point2f srcVertices[4];
    srcVertices[0] = Point(700, 605);
    srcVertices[1] = Point(890, 605);
    srcVertices[2] = Point(1760, 1030);
    srcVertices[3] = Point(20, 1030);

    Point2f dstVertices[4];
    dstVertices[0] = Point(0, 0);
    dstVertices[1] = Point(640, 0);
    dstVertices[2] = Point(640, 480);
    dstVertices[3] = Point(0, 480);

    Mat perspectiveMatrix = getPerspectiveTransform(srcVertices, dstVertices);
    Mat invertedPerspectiveMatrix;
    invert(perspectiveMatrix, invertedPerspectiveMatrix);
    // Apply the Perspective Transform just found to the src image
    warpPerspective(image, dst, perspectiveMatrix, dst.size(), INTER_LINEAR, BORDER_CONSTANT);
    
   
    //return output;
    return {dst, invertedPerspectiveMatrix};
}
vector<Point2f> LaneDetector::slidingWindow(Mat image, Rect window)
{
    vector<Point2f> points;
    const Size imgSize = image.size();
    bool shouldBreak = false;
    
    while (true)
    {
        float currentX = window.x + window.width * 0.5f;
        
        Mat roi = image(window); //Extract region of interest
        vector<Point2f> locations;
        
        findNonZero(roi, locations); //Get all non-black pixels. All are white in our case
        float avgX = 0.0f;
        
        for (int i = 0; i < locations.size(); ++i){
            float x = locations[i].x;
            avgX += window.x + x;
        }
        
        avgX = locations.empty() ? currentX : avgX / locations.size();
        
        Point point(avgX, window.y + window.height * 0.5f);
        points.push_back(point);
        
        //Move the window up
        window.y -= window.height;
        
        //For the uppermost position
        if (window.y < 0)
        {
            window.y = 0;
            shouldBreak = true;
        }
        
        //Move x position
        window.x += (point.x - currentX);
        
        //Make sure the window doesn't overflow, we get an error if we try to get data outside the matrix
        if (window.x < 0)
            window.x = 0;
        if (window.x + window.width >= imgSize.width)
            window.x = imgSize.width - window.width - 1;
        
        if (shouldBreak)
            break;
    }
    
    return points;
}
