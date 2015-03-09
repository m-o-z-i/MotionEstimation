#include "Visualisation.h"
#include "FindPoints.h"
#include "FindCameraMatrices.h"

inline static double square(int a)
{
    return a * a;
}

void drawAllStuff (cv::Mat mat_image11, cv::Mat mat_image12, cv::Mat mat_image21, cv::Mat mat_image22, int frame){
    vector<cv::Point2f> features1 = getStrongFeaturePoints(mat_image11, 150,0.01,5);
    drawPoints(mat_image11, features1, "1_left_features", cv::Scalar(0,0,0));

    pair<vector<cv::Point2f>, vector<cv::Point2f>> corresPoints1 = refindFeaturePoints(mat_image11, mat_image21, features1);
    drawPoints(mat_image12, corresPoints1.second, "1_corres points in right image", cv::Scalar(0,0,0));
    std::cout << "Frame: "<< frame << " found " << features1.size() << " features and " << corresPoints1.first.size() << "  corres Points " << std::endl;

    //pair<vector<cv::Point2f>, vector<cv::Point2f>> corresPoints2 = refindFeaturePoints(mat_image11, mat_image21, features1);
    //drawPoints(mat_image12, corresPoints2.second, "corresPoints in Frame21", cv::Scalar(0,255,255));

    //pair<vector<cv::Point2f>, vector<cv::Point2f>> corresPoints3 = refindFeaturePoints(mat_image11, mat_image22, features1);
    //drawPoints(mat_image12, corresPoints3.second, "corresPoints in Frame22", cv::Scalar(0,255,255));

    // get inliers from mean value
    vector<cv::Point2f> inliersM1, inliersM2;
    getInliersFromMeanValue(corresPoints1, &inliersM1, &inliersM2);
    std::cout << "deltete  " << corresPoints1.first.size() - inliersM1.size() << " outliers Points from mean value " << std::endl;
    drawPoints(mat_image12, inliersM2, "1_inliers by mean in right image", cv::Scalar(0,255,0));



    // get inliers from fundamental mat
    vector<cv::Point2f> inliersF1, inliersF2;
    cv::Mat F;
    getFundamentalMatrix(corresPoints1, &inliersF1, &inliersF2, F);
    drawEpipolarLines(mat_image11, mat_image12, inliersM1, inliersM2, F );
    std::cout << "deltete  " << corresPoints1.first.size() - inliersF1.size() << " outliers Points from fumdamentalmatrix " << std::endl;
    drawPoints(mat_image12, inliersF2, "1_inliers by fundamental in right image", cv::Scalar(255,255,0));

    //draw arrows
    drawCorresPoints(mat_image11, inliersF1, inliersF2, cv::Scalar(255,0,0) );


    cv::Mat flow, cflow;
    cv::calcOpticalFlowFarneback(mat_image11, mat_image21, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
    cv::cvtColor(mat_image11, cflow, CV_GRAY2BGR);
    drawOptFlowMap(flow, cflow, 200, CV_RGB(0, 255, 0));
    cv::imshow("optical flow field", cflow);
    cvWaitKey(0);
}


void drawOptFlowMap (cv::Mat flow, cv::Mat& cflowmap, int step, const cv::Scalar& color) {
    for(int y = 0; y < cflowmap.rows; y += step) {
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
            cv::line(cflowmap, cv::Point(x,y), cv::Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                     color);
            cv::circle(cflowmap, cv::Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), 1, color, -1);
        }
    }
}

void drawHomographyPoints(cv::Mat frame1, cv::Mat frame2, vector<cv::Point2f> const& points1, vector<cv::Point2f> const& points2){
    cv::Mat mat_color1;
    cv::Mat mat_color2;

    cv::cvtColor(frame1, mat_color1, CV_GRAY2RGB);
    cv::cvtColor(frame2, mat_color2, CV_GRAY2RGB);

    std::vector<uchar> inliers_homographie(points1.size(),0);
    cv::findHomography(cv::Mat(points1),cv::Mat(points2),inliers_homographie,CV_RANSAC,1.);
    // Draw the homography inlier points

    std::vector<cv::Point2f>::const_iterator itPts= points1.begin();
    std::vector<uchar>::const_iterator itIn= inliers_homographie.begin();

    cout << "Homography:  " << points1.size() << " " << points1.size() << endl;
    while (itPts!=points1.end()){

        // draw a circle at each inlier location
        if (*itIn)
            cv::circle(mat_color1,*itPts,3,cv::Scalar(0,255,0),2);
        else {
            cv::circle(mat_color1,*itPts,3,cv::Scalar(0,0,255),2);
        }

        ++itPts;
        ++itIn;
    }

    itPts= points2.begin();
    itIn= inliers_homographie.begin();
    while (itPts!=points2.end()) {

        // draw a circle at each inlier location
        if (*itIn)
            cv::circle(mat_color2,*itPts,3,cv::Scalar(0,255,0),2);
        else {
            cv::circle(mat_color2,*itPts,3,cv::Scalar(0,0,255),2);
        }

        ++itPts;
        ++itIn;
    }

    // Display the images with points
    cv::namedWindow("Right Image Homography (RANSAC)", cv::WINDOW_NORMAL);
    cv::imshow("Right Image Homography (RANSAC)",mat_color1);
    cv::namedWindow("Left Image Homography (RANSAC)", cv::WINDOW_NORMAL);
    cv::imshow("Left Image Homography (RANSAC)",mat_color2);

    // SAVE IMAGEs
    //string path = "data/image/epipoles/current"+(to_string(frame))+".png";
    //imwrite(path.c_str(), mat_image1);
    //cv::waitKey();
}

void drawEpipolarLines(cv::Mat frame1, cv::Mat frame2, const vector<cv::Point2f>& points1, const vector<cv::Point2f>& points2, cv::Mat F) {
    std::vector<uchar> inliers_fundamental(points1.size(),0);
    std::vector<cv::Vec3f> lines1;
    cv::computeCorrespondEpilines(cv::Mat(points1),1,F,lines1);
    for (vector<cv::Vec3f>::const_iterator it= lines1.begin();
         it!=lines1.end(); ++it) {

        cv::line(frame2,cv::Point(0,-(*it)[2]/(*it)[1]),
                cv::Point(frame2.cols,-((*it)[2]+(*it)[0]*frame2.cols)/(*it)[1]),
                cv::Scalar(255,255,255));
    }

    std::vector<cv::Vec3f> lines2;
    cv::computeCorrespondEpilines(cv::Mat(points2),2,F,lines2);
    for (vector<cv::Vec3f>::const_iterator it= lines2.begin();
         it!=lines2.end(); ++it) {

        cv::line(frame1,cv::Point(0,-(*it)[2]/(*it)[1]),
                cv::Point(frame1.cols,-((*it)[2]+(*it)[0]*frame1.cols)/(*it)[1]),
                cv::Scalar(255,255,255));
    }

    // Draw the inlier points
    std::vector<cv::Point2f> points1In, points2In;
    std::vector<cv::Point2f>::const_iterator itPts= points1.begin();
    std::vector<uchar>::const_iterator itIn= inliers_fundamental.begin();
    while (itPts!=points1.end()) {

        // draw a circle at each inlier location
        if (*itIn ) {
            cv::circle(frame1,*itPts,3,cv::Scalar(0,255,0),2);
            points1In.push_back(*itPts);
        }
        ++itPts;
        ++itIn;
    }

    itPts= points2.begin();
    itIn= inliers_fundamental.begin();
    while (itPts!=points2.end()) {

        // draw a circle at each inlier location
        if (*itIn) {
            cv::circle(frame2,*itPts,3,cv::Scalar(0,255,0),2);
            points2In.push_back(*itPts);
        }
        ++itPts;
        ++itIn;
    }

    // Display the images with points
    cv::imshow("Right Image Epilines (RANSAC)",frame1);
    cv::imshow("Left Image Epilines (RANSAC)",frame2);
}


void drawCorresPoints(cv::Mat image, vector<cv::Point2f> inliers1, vector<cv::Point2f> inliers2, cv::Scalar const& color) {
    // convert grayscale to color image
    cv::Mat color_image;
    cv::cvtColor(image, color_image, CV_GRAY2RGB);

    for(unsigned int i = 0; i < inliers1.size(); i++)
    {
        double angle;		angle = atan2( (double) inliers1[i].y - inliers2[i].y, (double) inliers1[i].x - inliers2[i].x );
        double hypotenuse;	hypotenuse = sqrt( square(inliers1[i].y - inliers2[i].y) + square(inliers1[i].x - inliers2[i].x) );

        /* Here we lengthen the arrow by a factor of three. */
        inliers2[i].x = (int) (inliers1[i].x - hypotenuse * cos(angle));
        inliers2[i].y = (int) (inliers1[i].y - hypotenuse * sin(angle));

        drawLine(color_image, inliers1[i], inliers2[i], angle, CV_RGB(color[0], color[1], color[2]));
    }

    /* Now display the image we drew on.  Recall that "Optical Flow" is the name of
     * the window we created above.
     */
    cv::imshow("OpticalFlow vectors", color_image);
}

void drawLine (cv::Mat ref, cv::Point2f p, cv::Point2f q, float angle, const cv::Scalar& color, int line_thickness ) {
    /* Now we draw the main line of the arrow. */
    /* "frame1" is the frame to draw on.
     * "p" is the point where the line begins.
     * "q" is the point where the line stops.
     * "CV_AA" means antialiased drawing.
     * "0" means no fractional bits in the center cooridinate or radius.
     */
    cv::line( ref, p, q, color, line_thickness, CV_AA, 0 );
    /* Now draw the tips of the arrow.  I do some scaling so that the
     * tips look proportional to the main line of the arrow.
     */
    p.x = (int) (q.x + 9 * cos(angle + pi / 4));
    p.y = (int) (q.y + 9 * sin(angle + pi / 4));
    cv::line( ref, p, q, color, line_thickness, CV_AA, 0 );
    p.x = (int) (q.x + 9 * cos(angle - pi / 4));
    p.y = (int) (q.y + 9 * sin(angle - pi / 4));
    cv::line( ref, p, q, color, line_thickness, CV_AA, 0 );
}

void drawLine (IplImage* ref, cv::Point2f p, cv::Point2f q, float angle, const cv::Scalar& color, int line_thickness ) {
    /* Now we draw the main line of the arrow. */
    /* "frame1" is the frame to draw on.
     * "p" is the point where the line begins.
     * "q" is the point where the line stops.
     * "CV_AA" means antialiased drawing.
     * "0" means no fractional bits in the center cooridinate or radius.
     */
    cvLine( ref, p, q, color, line_thickness, CV_AA, 0 );
    /* Now draw the tips of the arrow.  I do some scaling so that the
     * tips look proportional to the main line of the arrow.
     */
    p.x = (int) (q.x + 9 * cos(angle + pi / 4));
    p.y = (int) (q.y + 9 * sin(angle + pi / 4));
    cvLine( ref, p, q, color, line_thickness, CV_AA, 0 );
    p.x = (int) (q.x + 9 * cos(angle - pi / 4));
    p.y = (int) (q.y + 9 * sin(angle - pi / 4));
    cvLine( ref, p, q, color, line_thickness, CV_AA, 0 );
}

void drawPoints (cv::Mat image, vector<cv::Point2f> points, string windowName, cv::Scalar const& color) {
    cv::Mat colorImg;
    cv::cvtColor(image, colorImg, CV_GRAY2RGB);
    for (auto i : points) {
        // draw a circle at each inlier location
        cv::circle(colorImg,i,3,color,1);
    }
    cv::imshow(windowName, colorImg);
}
