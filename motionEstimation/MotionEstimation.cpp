#include "bgapi2_genicam.hpp"
#include <opencv2/opencv.hpp>

#include "line/MyLine.h"

#include <cmath>
#include <math.h> 
#include <vector>
#include <utility>

#include <sstream>
#include <string.h>

#include <opencv2/core/core.hpp>

using namespace std;

static const double pi = 3.14159265358979323846;

inline static double square(int a)
{
	return a * a;
}

char key;

int 					m_camWidth;
int 					m_camHeight;

CvSize 					m_imageSize(cvSize(1384, 1036));
IplImage* 				m_frame1(cvCreateImage(m_imageSize, IPL_DEPTH_8U, 1));

BGAPI2::SystemList* 	m_systemList(NULL);
BGAPI2::System* 		m_systemMem(NULL);
BGAPI2::String 			m_systemID("");

BGAPI2::InterfaceList*	m_interfaceList(NULL);
BGAPI2::Interface* 		m_interface(NULL);
BGAPI2::String			m_interfaceID("");

BGAPI2::DeviceList*		m_deviceList(NULL);
BGAPI2::Device*			m_device(NULL);
BGAPI2::String			m_deviceID("");

BGAPI2::DataStreamList*	m_datastreamList(NULL);
BGAPI2::DataStream*		m_datastream(NULL);
BGAPI2::String			m_datastreamID("");

BGAPI2::BufferList*		m_bufferList(NULL);
BGAPI2::Buffer*			m_buffer(NULL);


// callbacks
void init_camera();
void open_stream(IplImage* ref);
void feature_tracking(IplImage* image1, IplImage* image2, int frame);
void epipole_tracking(IplImage* image1, IplImage* image2, int frame);
void drawLine(IplImage* ref, cv::Point2f p, cv::Point2f q, float angle, CvScalar const& color = CV_RGB(0,0,0), int line_thickness = 1);
void drawLine(cv::Mat ref, cv::Point2f p, cv::Point2f q, float angle, CvScalar const& color = CV_RGB(0,0,0), int line_thickness = 1);

std::vector<cv::Point2f> getStrongFeaturePoints (cv::Mat image, int number = 50, float qualityLevel = .03, float minDistance = 0.1);
pair<vector<cv::Point2f>, vector<cv::Point2f> > refindFeaturePoints(cv::Mat prev_image, cv::Mat next_image, vector<cv::Point2f> frame1_features);
void drawEpipolarLines(cv::Mat frame1, cv::Mat frame2, vector<cv::Point2f> points1, vector<cv::Point2f> points2);
void getInliers (pair<vector<cv::Point2f>, vector<cv::Point2f> > features, vector<cv::Point2f> *inliers2, vector<cv::Point2f> *inliers1);
void drawCorresPoints(cv::Mat image, vector<cv::Point2f> inliers1, vector<cv::Point2f> inliers2, const CvScalar& color, int id);

//TODO: calcOpticalFlowFarneback
/* STEP BY STEP:
 * 1.  capture stereo calibrated images in frame 1
 * 2.1 find feature points in image 1.1
 * 2.2 find corresponding points in image 1.2
 * 3.  triangulate 3d points from frame 1
 * 4.  wait one frame
 * 5.  capture again images from frame 2
 * 6.  try to find same corresponding points from frame 1 in new stereo images from frame 2
 * 7.  triangulate 3d points from frame 2
 * 8.  calculate essential matrix from frame 1 to frame 2
 * 9.  estimate motion with 2 sets of 2d and 3d points and the essential matrix .. how?
 * 10. swap 2d points of frame 1 and frame 2
 * 11. try to add some new feature points (until defined numer of necessary points are reached)
 * 12. continue with step 4
 */


int main() {

    // initialize baumer camera
	init_camera();

    // init images
	IplImage* image1 = cvCreateImage(m_imageSize, IPL_DEPTH_8U, 1);
	IplImage* image2 = cvCreateImage(m_imageSize, IPL_DEPTH_8U, 1);

	/*	
	while (true) {
		open_stream(image1);
		cvShowImage("Optical Flow", image1);
		key = cvWaitKey(10);
	}
	*/

    int frame=1;

	while(true)
	{
        //open_stream(image1);
        //key = cvWaitKey(1);
        //open_stream(image2);

		//convert to Mat
        //cv::Mat mat_image1(image1);
        //cv::Mat mat_image2(image2);

        //stereo1
        cv::Mat mat_image11 = cv::imread("data/stereoImages/left/"+(std::to_string(frame))+"_l.jpg",0);
        cv::Mat mat_image12 = cv::imread("data/stereoImages/right/"+(std::to_string(frame))+"_r.jpg",0);

        //stereo2
        cv::Mat mat_image21 = cv::imread("data/stereoImages/left/"+(std::to_string(frame+1))+"_l.jpg",0);
        cv::Mat mat_image22 = cv::imread("data/stereoImages/right/"+(std::to_string(frame+1))+"_r.jpg",0);

        cv::imshow("11", mat_image11);
        cv::imshow("12", mat_image12);
        cv::imshow("21", mat_image21);
        cv::imshow("22", mat_image22);

        if(! mat_image11.data || !mat_image12.data || !mat_image22.data || !mat_image21.data)                              // Check for invalid input
        {
            cout <<  "Could not open or find the image: "  << std::endl ;
            break;
        }

        vector<cv::Point2f> features1 = getStrongFeaturePoints(mat_image11);
        pair<vector<cv::Point2f>, vector<cv::Point2f>> corresPoints1 = refindFeaturePoints(mat_image11, mat_image12, features1);
        pair<vector<cv::Point2f>, vector<cv::Point2f>> corresPoints2 = refindFeaturePoints(mat_image11, mat_image21, features1);
        pair<vector<cv::Point2f>, vector<cv::Point2f>> corresPoints3 = refindFeaturePoints(mat_image11, mat_image22, features1);

        drawCorresPoints(mat_image11,corresPoints1.first, corresPoints1.second, cvScalar(255,0,0),1);
        drawCorresPoints(mat_image11,corresPoints2.first, corresPoints2.second, cvScalar(0,255,0),2);
        drawCorresPoints(mat_image11,corresPoints3.first, corresPoints3.second, cvScalar(0,0,255),3);

        //vector<cv::Point2f> inliers1, inliers2;
        //getInliers(corresPoints, &inliers1, &inliers2);
        //std::cout << "main:   " << inliers1.size() << "  " << inliers2.size() << std::endl;

        //drawEpipolarLines(mat_image1, mat_image2, inliers1, inliers2);


        ++frame;
        cvWaitKey();
	}

	return 0;
}

vector<cv::Point2f> getStrongFeaturePoints(cv::Mat image, int number, float qualityLevel, float minDistance) {
    /* Shi and Tomasi Feature Tracking! */

    /* Preparation: This array will contain the features found in image1. */
    vector<cv::Point2f> image_features;

    /* Preparation: BEFORE the function call this variable is the array size
     * (or the maximum number of features to find).  AFTER the function call
     * this variable is the number of features actually found.
     */
    int number_of_features = number;

    /* Actually run the Shi and Tomasi algorithm!!
     * "Image" is the input image.
     * qualityLevel specifies the minimum quality of the features (based on the eigenvalues)
     * minDistance specifies the minimum Euclidean distance between features
     * RETURNS:
     * "image_features" will contain the feature points.
     */
    cv::goodFeaturesToTrack(image, image_features, number_of_features, qualityLevel, minDistance);

    return image_features;
}

pair<vector<cv::Point2f>, vector<cv::Point2f>> refindFeaturePoints(cv::Mat prev_image, cv::Mat next_image, vector<cv::Point2f> frame1_features){
    /* Pyramidal Lucas Kanade Optical Flow! */

    /* This array will contain the locations of the points from frame 1 in frame 2. */
    vector<cv::Point2f>  frame2_features;

    /* The i-th element of this array will be non-zero if and only if the i-th feature of
     * frame 1 was found in frame 2.
     */
    vector<unsigned char> optical_flow_found_feature;

    /* The i-th element of this array is the error in the optical flow for the i-th feature
     * of frame1 as found in frame 2.  If the i-th feature was not found (see the array above)
     * I think the i-th entry in this array is undefined.
     */
    vector<float> optical_flow_feature_error;

    /* This is the window size to use to avoid the aperture problem (see slide "Optical Flow: Overview"). */
    CvSize optical_flow_window = cvSize(3,3);

    /* 0-based maximal pyramid level number; if set to 0, pyramids are not used (single level),
     * if set to 1, two levels are used, and so on; if pyramids are passed to input then algorithm
     * will use as many levels as pyramids have but no more than maxLevel.
     * */
    int maxLevel = 5;

    /* This termination criteria tells the algorithm to stop when it has either done 20 iterations or when
     * epsilon is better than .3.  You can play with these parameters for speed vs. accuracy but these values
     * work pretty well in many situations.
     */
    cv::TermCriteria optical_flow_termination_criteria
        = cv::TermCriteria( cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, .3 );

    /* Actually run Pyramidal Lucas Kanade Optical Flow!!
     * "prev_image" is the first frame with the known features. pyramid constructed by buildOpticalFlowPyramid()
     * "next_image" is the second frame where we want to find the first frame's features.
     * "frame1_features" are the features from the first frame.
     * "frame2_features" is the (outputted) locations of those features in the second frame.
     * "number_of_features" is the number of features in the frame1_features array.
     * "optical_flow_window" is the size of the window to use to avoid the aperture problem.
     * "maxLevel" is the maximum number of pyramids to use.  0 would be just one level.
     * "optical_flow_found_feature" is as described above (non-zero iff feature found by the flow).
     * "optical_flow_feature_error" is as described above (error in the flow for this feature).
     * "optical_flow_termination_criteria" is as described above (how long the algorithm should look).
     * "0" means disable enhancements.  (For example, the second array isn't pre-initialized with guesses.)
     */
    //TODO: improve TermCriteria. do not quit program when it is reached
    cv::calcOpticalFlowPyrLK(prev_image, next_image, frame1_features, frame2_features, optical_flow_found_feature,
                             optical_flow_feature_error, optical_flow_window, maxLevel,
                             optical_flow_termination_criteria, cv::OPTFLOW_LK_GET_MIN_EIGENVALS);


    vector<cv::Point2f>::iterator iter_f1 = frame1_features.begin();
    vector<cv::Point2f>::iterator iter_f2 = frame2_features.begin();
    for (unsigned i = 0; i < frame1_features.size(); ++i){
        if ( optical_flow_found_feature[i] == 0 ){
            frame1_features.erase(iter_f1);
            frame2_features.erase(iter_f2);
        }
        ++iter_f1;
        ++iter_f2;
    }

    return make_pair(frame1_features, frame2_features);
}

void getInliers (pair<vector<cv::Point2f>, vector<cv::Point2f>> features, vector<cv::Point2f>* inliers1, vector<cv::Point2f>* inliers2){
    vector<double> directions;
    vector<double> lengths;

    for (unsigned i = 0; i < features.first.size(); ++i){
        double direction = atan2( (double) features.first[i].y - features.second[i].y, (double) features.second[i].x - features.second[i].x );
        directions.push_back(direction);

        double length = sqrt( square(features.first[i].y - features.second[i].y) + square(features.first[i].x - features.second[i].x) );
        lengths.push_back(length);
    }

    sort(directions.begin(), directions.end());
    double median_angle = directions[(int)(directions.size()/2)];

    sort(lengths.begin(),lengths.end());
    double median_lenght = lengths[(int)(lengths.size()/2)];

    for(unsigned i = 0; i < features.first.size(); ++i)
    {
        double direction = atan2( (double) features.first[i].y - features.second[i].y, (double) features.second[i].x - features.second[i].x );
        double length = sqrt( square(features.first[i].y - features.second[i].y) + square(features.first[i].x - features.second[i].x) );

        if (direction < median_angle + 2 && direction > median_angle - 2 ) {
            if (length < (median_lenght*3) && length > 1.5 && length > median_lenght*0.1) {
                inliers1->push_back(features.first[i]);
                inliers2->push_back(features.second[i]);
            }
        }
    }
}

void drawEpipolarLines(cv::Mat frame1, cv::Mat frame2, vector<cv::Point2f> points1, vector<cv::Point2f> points2) {
    cv::Mat mat_color1;
    cv::Mat mat_color2;

    cv::cvtColor(frame1, mat_color1, CV_GRAY2RGB);
    cv::cvtColor(frame2, mat_color2, CV_GRAY2RGB);

    // Compute F matrix using RANSAC
    if (points1.size()>10 && points2.size()>10){
        std::cout << "fundamental: " << points1.size() << " " << points2.size() << " " << cv::Mat(points1).rows << std::endl;
        std::vector<uchar> inliers_fundamental(points1.size(),0);
        cv::Mat fundemental = cv::findFundamentalMat(
                              cv::Mat(points1), cv::Mat(points2),   // matching points
                              inliers_fundamental,                  // match status (inlier ou outlier)
                              CV_FM_LMEDS,                          // RANSAC method
                              1,                                    // distance to epipolar line
                              0.98);                                // confidence probability

        std::vector<cv::Vec3f> lines1;
        cv::computeCorrespondEpilines(cv::Mat(points1),1,fundemental,lines1);
        for (vector<cv::Vec3f>::const_iterator it= lines1.begin();
             it!=lines1.end(); ++it) {

                 cv::line(frame2,cv::Point(0,-(*it)[2]/(*it)[1]),
                                 cv::Point(frame2.cols,-((*it)[2]+(*it)[0]*frame2.cols)/(*it)[1]),
                                 cv::Scalar(255,255,255));
        }

        std::vector<cv::Vec3f> lines2;
        cv::computeCorrespondEpilines(cv::Mat(points2),2,fundemental,lines2);
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
                cv::circle(frame1,*itPts,3,cv::Scalar(255,255,255),2);
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
                cv::circle(frame2,*itPts,3,cv::Scalar(255,255,255),2);
                points2In.push_back(*itPts);
            }
            ++itPts;
            ++itIn;
        }

        // Display the images with points
        cv::namedWindow("Right Image Epilines (RANSAC)", cv::WINDOW_NORMAL);
        cv::imshow("Right Image Epilines (RANSAC)",frame1);
        cv::namedWindow("Left Image Epilines (RANSAC)", cv::WINDOW_NORMAL);
        cv::imshow("Left Image Epilines (RANSAC)",frame2);


        std::vector<uchar> inliers_homographie(points1.size(),0);
        cv::findHomography(cv::Mat(points1In),cv::Mat(points2In),inliers_homographie,CV_RANSAC,1.);
        // Draw the homography inlier points
        itPts= points1In.begin();
        itIn= inliers_homographie.begin();
        cout << "Homography:  " << points1In.size() << " " << points2In.size() << endl;
        while (itPts!=points1In.end()) {

            // draw a circle at each inlier location
            if (*itIn)
                cv::circle(mat_color1,*itPts,3,cv::Scalar(0,255,0),2);
            else {
                cv::circle(mat_color1,*itPts,3,cv::Scalar(0,0,255),2);
            }

            ++itPts;
            ++itIn;
        }

        itPts= points2In.begin();
        itIn= inliers_homographie.begin();
        while (itPts!=points2In.end()) {

            // draw a circle at each inlier location
            if (*itIn)
                cv::circle(mat_color2,*itPts,3,cv::Scalar(0,255,0),2);
            else {
                cv::circle(mat_color2,*itPts,3,cv::Scalar(0,0,255),2);
            }

            ++itPts;
            ++itIn;
        }

        if (inliers_homographie.size() > inliers_fundamental.size()){
            cout << "skip frame because points are on one plane" << endl;
        }

        // Display the images with points
        cv::namedWindow("Right Image Homography (RANSAC)", cv::WINDOW_NORMAL);
        cv::imshow("Right Image Homography (RANSAC)",mat_color1);
        cv::namedWindow("Left Image Homography (RANSAC)", cv::WINDOW_NORMAL);
        cv::imshow("Left Image Homography (RANSAC)",mat_color2);

        // SAVE IMAGEs
        //string path = "data/image/epipoles/current"+(to_string(frame))+".png";
        //imwrite(path.c_str(), mat_image1);
        cv::waitKey();
    }
}

void drawCorresPoints(cv::Mat image, vector<cv::Point2f> inliers1, vector<cv::Point2f> inliers2, CvScalar const& color, int id) {
    // convert grayscale to color image
    cv::Mat color_image;
    cv::cvtColor(image, color_image, CV_GRAY2RGB);

    vector<double> directions, lengths;

    for (unsigned i = 0; i < inliers1.size(); ++i){
        double direction = atan2( (double) inliers1[i].y - inliers2[i].y, (double) inliers2[i].x - inliers2[i].x );
        directions.push_back(direction);

        double length = sqrt( square(inliers1[i].y - inliers2[i].y) + square(inliers1[i].x - inliers2[i].x) );
        lengths.push_back(length);
    }

    sort(directions.begin(), directions.end());
    double median_angle = directions[(int)(directions.size()/2)];

    sort(lengths.begin(),lengths.end());
    double median_lenght = lengths[(int)(lengths.size()/2)];


    for(int i = 0; i < inliers1.size(); i++)
    {
        double angle;		angle = atan2( (double) inliers1[i].y - inliers2[i].y, (double) inliers1[i].x - inliers2[i].x );
        double hypotenuse;	hypotenuse = sqrt( square(inliers1[i].y - inliers2[i].y) + square(inliers1[i].x - inliers2[i].x) );

        /* Here we lengthen the arrow by a factor of three. */
        inliers2[i].x = (int) (inliers1[i].x - 3 * hypotenuse * cos(angle));
        inliers2[i].y = (int) (inliers1[i].y - 3 * hypotenuse * sin(angle));

        if (angle < median_angle + 2 && angle > median_angle - 2 ) {
            if (hypotenuse < (median_lenght*3) && hypotenuse > 1.5 && hypotenuse > median_lenght*0.1) {
                drawLine(color_image, inliers1[i], inliers2[i], angle, CV_RGB(color.val[0], color.val[1], color.val[2]));
            } else {
                drawLine(color_image, inliers1[i], inliers2[i], angle, CV_RGB(0,0,0));
            }
        } else {
            drawLine(color_image, inliers1[i], inliers2[i], angle, CV_RGB(0,0,0));
        }
    }


    /* Now display the image we drew on.  Recall that "Optical Flow" is the name of
     * the window we created above.
     */
    cv::imshow("Optical Flow"+to_string(id), color_image);

    // save image in every frame
    //string path = "data/image/vectors/current"+(to_string(frame))+".png";
    //cvSaveImage(path.c_str(), colorImage);
}

void drawLine (cv::Mat ref, cv::Point2f p, cv::Point2f q, float angle, CvScalar const& color, int line_thickness ) {
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

void drawLine (IplImage* ref, cv::Point2f p, cv::Point2f q, float angle, CvScalar const& color, int line_thickness ) {
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

void init_camera() {
    // SYSTEM



    try {
        m_systemList = BGAPI2::SystemList::GetInstance();
        m_systemList->Refresh();
        std::cout << "Detected systems: " << m_systemList->size() << std::endl;

        m_systemList->begin()->second->Open();
        m_systemID = m_systemList->begin()->first;
        if(m_systemID == "") {
            std::cout << "Error: no system found" << std::endl;
        }
        else {
            m_systemMem = (*m_systemList)[m_systemID];
            std::cout << "SystemID:  " << m_systemID << std::endl;
        }

        //INTERFACE
        m_interfaceList = m_systemMem->GetInterfaces();
        m_interfaceList->Refresh(100);
        std::cout << "Detected interfaces: " << m_interfaceList->size() << std::endl;

        for (BGAPI2::InterfaceList::iterator interfaceIter = m_interfaceList->begin(); interfaceIter != m_interfaceList->end(); interfaceIter++) {
            interfaceIter->second->Open();
            m_deviceList = interfaceIter->second->GetDevices();
            m_deviceList->Refresh(100);

            if (m_deviceList->size() > 0) {
                std::cout << "Detected Devices: " << m_deviceList->size() << std::endl;
                m_interfaceID = interfaceIter->first;
                m_interface = interfaceIter->second;
                break;
            }
            else {
                interfaceIter->second->Close();
            }
        }

        // DEVICE
        m_device  = m_deviceList->begin()->second;
        m_device->Open();
        m_deviceID = m_deviceList->begin()->first;
        if(m_deviceID == "") {
            std::cout << "Error: no camera found" << std::endl;
        }
        else {
            m_device = (*m_deviceList)[m_deviceID];
            std::cout << "DeviceID: " << m_deviceID << std::endl;
        }

        m_device->GetRemoteNode("PixelFormat")->SetString("Mono8");
        m_device->GetRemoteNode("Gain")->SetDouble(10.00);
        //m_device->GetRemoteNode("TriggerMode")->SetString("On");
        //m_device->GetRemoteNode("TriggerSource")->SetValue("Line0");
        //m_device->GetRemoteNode("ExposureTime")->SetDouble(13000);

        //Set cam resolution to halve reolution [696, 520]
        //std::cout << "1 " << m_device->GetRemoteNode("TriggerSource")->GetDescription()  << std::endl;
        //std::cout << "2 " << m_device->GetRemoteNode("TriggerSource")->GetInterface()  << std::endl;
        //m_device->GetRemoteNode("BinningHorizontal")->SetInt( 2);
        //m_device->GetRemoteNode("BinningVertical")->SetInt( 2);


        // GET CAM RESOLUTION
        m_camWidth = m_device->GetRemoteNode("Width")->GetInt();
        m_camHeight = m_device->GetRemoteNode("Height")->GetInt();
        std::cout << "Cam resolution : " << m_camWidth << "  " <<  m_camHeight << std::endl;


        // DATASTREAM
        m_datastreamList = m_device->GetDataStreams();
        m_datastreamList->Refresh();
        std::cout << "Detected datastreams: " << m_datastreamList->size() << std::endl;

        m_datastreamList->begin()->second->Open();
        m_datastreamID = m_datastreamList->begin()->first;
        if(m_datastreamID == "") {
            std::cout << "Error: no datastream found" << std::endl;
        }
        else{
            m_datastream = (*m_datastreamList)[m_datastreamID];
            std::cout << "DatastreamID: " << m_datastreamID << std::endl;
        }

        // BUFFER
        m_bufferList = m_datastream->GetBufferList();
        for(int i=0; i<(4); i++) { // 4 buffers using internal buffers
             m_buffer = new BGAPI2::Buffer();
             m_bufferList->Add(m_buffer);
        }
        std::cout << "Announced buffers: " << m_bufferList->size() << std::endl;
        for (BGAPI2::BufferList::iterator buf = m_bufferList->begin(); buf != m_bufferList->end(); buf++) {
             buf->second->QueueBuffer();
        }
        std::cout << "Queued buffers: " << m_bufferList->GetQueuedCount() << std::endl;

        // START DATASTREAM AND FILL BUFFER
        m_datastream->StartAcquisitionContinuous();
        m_device->GetRemoteNode("AcquisitionStart")->Execute();

    } catch (BGAPI2::Exceptions::IException& ex) {
        std::cerr << ex.GetErrorDescription() << std::endl;
    }

}

void open_stream(IplImage* ref) {
    char* img = nullptr;
    try {
        BGAPI2::Buffer* m_bufferFilled = NULL;
        m_bufferFilled = m_datastream->GetFilledBuffer(1000);
        if(m_bufferFilled == NULL){
            std::cout << "Error: buffer timeout" << std::endl;
        }

        img = (char*)m_bufferFilled->GetMemPtr();
        m_bufferFilled->QueueBuffer();

        IplImage* frameTemp = cvCreateImageHeader(cvSize(m_camWidth, m_camHeight), IPL_DEPTH_8U, 1);
        cvSetData(frameTemp, img, m_camWidth);

        cvCopy(frameTemp, ref, NULL);
        cvReleaseImageHeader(&frameTemp);

        cvFlip(ref, ref, -1);

        if (char(key) == 32) { // Space saves the current image
            cvSaveImage("current.png", ref);
        }

    } catch (BGAPI2::Exceptions::IException& ex) {
        std::cerr << ex.GetErrorDescription() << std::endl;
    }

}
