#include "bgapi2_genicam.hpp"
#include <opencv2/opencv.hpp>

#include "myline.h"

#include <cmath>
#include <math.h> 
#include <vector>
#include <utility>

#include <sstream>
#include <string.h>

using namespace cv;
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
void feature_tracking(IplImage* image1, IplImage* image2);
void epipole_tracking(IplImage* image1, IplImage* image2);
void drawLine(IplImage* ref, Point2f p, Point2f q, float angle, CvScalar const& color = CV_RGB(0,0,0), int line_thickness = 1);

int main() {
	// initialize baumer camera
	init_camera();

	IplImage* image1 = cvCreateImage(m_imageSize, IPL_DEPTH_8U, 1);
	IplImage* image2 = cvCreateImage(m_imageSize, IPL_DEPTH_8U, 1);

	/*	
	while (true) {
		open_stream(image1);
		cvShowImage("Optical Flow", image1);
		key = cvWaitKey(10);
	}
	*/

	int frame=0;

	cvNamedWindow("Optical Flow", CV_WINDOW_AUTOSIZE);


	while(true)
	{

		++frame; 

		open_stream(image1);
		key = cvWaitKey(1);
		open_stream(image2);

		//convert to Mat
		//cv::Mat mat_image(image1);
		//imshow("Mat", mat_image);

		epipole_tracking(image1, image2);
	}

	return 0;
}



void epipole_tracking(IplImage* image1, IplImage* image2) {
	std::vector<double> lengths1;
	std::vector<double> lengths2;
	std::vector<double> directions;

	IplImage* colorImage = cvCreateImage(m_imageSize, IPL_DEPTH_8U, 3);

	/* Shi and Tomasi Feature Tracking! */

	/* Preparation: Allocate the necessary storage. */
	IplImage* eig_image = cvCreateImage(m_imageSize, IPL_DEPTH_8U, 1 );
	IplImage* temp_image = cvCreateImage(m_imageSize, IPL_DEPTH_8U, 1 );

	/* Preparation: This array will contain the features found in frame 1. */
	CvPoint2D32f frame1_features[50];
	/* Preparation: BEFORE the function call this variable is the array size
	 * (or the maximum number of features to find).  AFTER the function call
	 * this variable is the number of features actually found.
	 */
	int number_of_features;

	/* I'm hardcoding this at 50.  But you should make this a #define so that you can
	 * change the number of features you use for an accuracy/speed tradeoff analysis.
	 */
	number_of_features = 50;

	/* Actually run the Shi and Tomasi algorithm!!
	 * "frame1_1C" is the input image.
	 * "eig_image" and "temp_image" are just workspace for the algorithm.
	 * The first ".01" specifies the minimum quality of the features (based on the eigenvalues).
	 * The second ".01" specifies the minimum Euclidean distance between features.
	 * "NULL" means use the entire input image.  You could point to a part of the image.
	 * WHEN THE ALGORITHM RETURNS:
	 * "frame1_features" will contain the feature points.
	 * "number_of_features" will be set to a value <= 50 indicating the number of feature points found.
	 */
	cvGoodFeaturesToTrack(image1, eig_image, temp_image, frame1_features, &number_of_features, .03, .1, NULL);


	/* Pyramidal Lucas Kanade Optical Flow! */

	/* This array will contain the locations of the points from frame 1 in frame 2. */
	CvPoint2D32f frame2_features[50];
	/* The i-th element of this array will be non-zero if and only if the i-th feature of
	 * frame 1 was found in frame 2.
	 */
	char optical_flow_found_feature[50];
	/* The i-th element of this array is the error in the optical flow for the i-th feature
	 * of frame1 as found in frame 2.  If the i-th feature was not found (see the array above)
	 * I think the i-th entry in this array is undefined.
	 */
	float optical_flow_feature_error[50];

	/* This is the window size to use to avoid the aperture problem (see slide "Optical Flow: Overview"). */
	CvSize optical_flow_window = cvSize(3,3);

	/* This termination criteria tells the algorithm to stop when it has either done 20 iterations or when
	 * epsilon is better than .3.  You can play with these parameters for speed vs. accuracy but these values
	 * work pretty well in many situations.
	 */
	CvTermCriteria optical_flow_termination_criteria
		= cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 30, .3 );

	/* This is some workspace for the algorithm.
	 * (The algorithm actually carves the image into pyramids of different resolutions.)
	 */
	IplImage* pyramid1 = cvCreateImage( m_imageSize, IPL_DEPTH_8U, 1 );
	IplImage* pyramid2 = cvCreateImage( m_imageSize, IPL_DEPTH_8U, 1 );

	/* Actually run Pyramidal Lucas Kanade Optical Flow!!
	 * "frame1_1C" is the first frame with the known features.
	 * "frame2_1C" is the second frame where we want to find the first frame's features.
	 * "pyramid1" and "pyramid2" are workspace for the algorithm.
	 * "frame1_features" are the features from the first frame.
	 * "frame2_features" is the (outputted) locations of those features in the second frame.
	 * "number_of_features" is the number of features in the frame1_features array.
	 * "optical_flow_window" is the size of the window to use to avoid the aperture problem.
	 * "5" is the maximum number of pyramids to use.  0 would be just one level.
	 * "optical_flow_found_feature" is as described above (non-zero iff feature found by the flow).
	 * "optical_flow_feature_error" is as described above (error in the flow for this feature).
	 * "optical_flow_termination_criteria" is as described above (how long the algorithm should look).
	 * "0" means disable enhancements.  (For example, the second array isn't pre-initialized with guesses.)
	 */
	cvCalcOpticalFlowPyrLK(image1, image2, pyramid1, pyramid2, frame1_features, frame2_features, number_of_features, 
	 					   optical_flow_window, 5, optical_flow_found_feature, optical_flow_feature_error, 
	 					   optical_flow_termination_criteria, OPTFLOW_LK_GET_MIN_EIGENVALS);
		
	cv::Mat mat_image1(image1);
	cv::Mat mat_image2(image2);
	cv::Mat mat_color1;
	cv::Mat mat_color2;

	cv::cvtColor(mat_image1, mat_color1, CV_GRAY2RGB);
	cv::cvtColor(mat_image2, mat_color2, CV_GRAY2RGB);


	//get median angle and length
	vector<myLine>corresPoints;
	for(int i = 0; i < number_of_features; i++)
	{
		if ( optical_flow_found_feature[i] == 0 )	continue;

		Point2f a,b;
		a.x = (int) frame1_features[i].x;
		a.y = (int) frame1_features[i].y;
		b.x = (int) frame2_features[i].x;
		b.y = (int) frame2_features[i].y;

		corresPoints.push_back(myLine(a,b));
	}
	sort(corresPoints.begin(), corresPoints.end(),[](myLine a, myLine b) -> bool { return a.getLength() > b.getLength();});
	double median_lenght = corresPoints[(int)(corresPoints.size()/2)].getLength();
	
	sort(corresPoints.begin(), corresPoints.end(),[](myLine a, myLine b) -> bool { return a.getAngle() > b.getAngle();});
	double median_angle = corresPoints[(int)(corresPoints.size()/2)].getAngle();


	// Convert inliers into Point2f
	std::vector<cv::Point2f> points1, points2;
	for(auto i = corresPoints.begin(); i < corresPoints.end(); ++i)
	{
		if (i->getAngle() < median_angle + 2 && i->getAngle() > median_angle - 2 ) {
			if (i->getLength() < (median_lenght*3) && i->getLength() > 1.5 && i->getLength() > median_lenght*0.1) {
				points1.push_back(i->getPointA());
				points2.push_back(i->getPointB());
			}
		}
	}

	std::cout << points1.size() << " " << points2.size() << " " << cv::Mat(points1).rows << std::endl; 

	// Compute F matrix using RANSAC
	std::vector<uchar> inliers(points1.size(),0);
	if (points1.size()>10 && points2.size()>10){
		cv::Mat fundemental = cv::findFundamentalMat(
								cv::Mat(points1), cv::Mat(points2), // matching points
								inliers,      // match status (inlier ou outlier)  
								CV_FM_RANSAC, // RANSAC method
								1,            // distance to epipolar line
								0.98);        // confidence probability

		std::vector<cv::Vec3f> lines1; 
		cv::computeCorrespondEpilines(cv::Mat(points1),1,fundemental,lines1);
		for (vector<cv::Vec3f>::const_iterator it= lines1.begin();
			 it!=lines1.end(); ++it) {

				 cv::line(mat_image2,cv::Point(0,-(*it)[2]/(*it)[1]),
					             cv::Point(mat_image2.cols,-((*it)[2]+(*it)[0]*mat_image2.cols)/(*it)[1]),
								 cv::Scalar(255,255,255));
		}
		
		std::vector<cv::Vec3f> lines2; 
		cv::computeCorrespondEpilines(cv::Mat(points2),2,fundemental,lines2);
		for (vector<cv::Vec3f>::const_iterator it= lines2.begin();
			 it!=lines2.end(); ++it) {

				 cv::line(mat_image1,cv::Point(0,-(*it)[2]/(*it)[1]),
					             cv::Point(mat_image1.cols,-((*it)[2]+(*it)[0]*mat_image1.cols)/(*it)[1]),
								 cv::Scalar(255,255,255));
		}

		// Draw the inlier points
		std::vector<cv::Point2f> points1In, points2In;
		std::vector<cv::Point2f>::const_iterator itPts= points1.begin();
		std::vector<uchar>::const_iterator itIn= inliers.begin();
		while (itPts!=points1.end()) {

			// draw a circle at each inlier location
			if (*itIn) {
	 			cv::circle(mat_image1,*itPts,3,cv::Scalar(255,255,255),2);
				points1In.push_back(*itPts);
			}
			++itPts;
			++itIn;
		}

		itPts= points2.begin();
		itIn= inliers.begin();
		while (itPts!=points2.end()) {

			// draw a circle at each inlier location
			if (*itIn) {
				cv::circle(mat_image2,*itPts,3,cv::Scalar(255,255,255),2);
				points2In.push_back(*itPts);
			}
			++itPts;
			++itIn;
		}

		// Display the images with points
		cv::namedWindow("Right Image Epilines (RANSAC)", WINDOW_NORMAL);
		cv::imshow("Right Image Epilines (RANSAC)",mat_image1);
		cv::namedWindow("Left Image Epilines (RANSAC)", WINDOW_NORMAL);
		cv::imshow("Left Image Epilines (RANSAC)",mat_image2);

		if (points1In.size() > 0 && points2In.size() > 0) {
			cv::findHomography(cv::Mat(points1In),cv::Mat(points2In),inliers,CV_RANSAC,1.);
			// Draw the inlier points
			itPts= points1In.begin();
			itIn= inliers.begin();
			cout << "go" << points1In.size() << " " << points2In.size() << endl;
			while (itPts!=points1In.end()) {

				// draw a circle at each inlier location
				if (*itIn) 
		 			cv::circle(mat_color1,*itPts,3,cv::Scalar(255,0,0),2);
		 		else {
		 			cv::circle(mat_color1,*itPts,3,cv::Scalar(0,0,255),2);
		 		}
				
				++itPts;
				++itIn;
			}

			itPts= points2In.begin();
			itIn= inliers.begin();
			while (itPts!=points2In.end()) {

				// draw a circle at each inlier location
				if (*itIn) 
					cv::circle(mat_color2,*itPts,3,cv::Scalar(255,0,0),2);
				else {
		 			cv::circle(mat_color2,*itPts,3,cv::Scalar(0,0,255),2);
		 		}

				++itPts;
				++itIn;
			}

		    // Display the images with points
			cv::namedWindow("Right Image Homography (RANSAC)", WINDOW_NORMAL);
			cv::imshow("Right Image Homography (RANSAC)",mat_color1);
			cv::namedWindow("Left Image Homography (RANSAC)", WINDOW_NORMAL);
			cv::imshow("Left Image Homography (RANSAC)",mat_color2);

			cv::waitKey();
		}
		mat_image1.release();
		mat_image2.release();
		mat_color1.release();
		mat_color2.release();

	}
}


void feature_tracking(IplImage* image1, IplImage* image2) {
		std::vector<double> lengths1;
		std::vector<double> lengths2;
		std::vector<double> directions;

		IplImage* colorImage = cvCreateImage(m_imageSize, IPL_DEPTH_8U, 3);

		/* Shi and Tomasi Feature Tracking! */

		/* Preparation: Allocate the necessary storage. */
		IplImage* eig_image = cvCreateImage(m_imageSize, IPL_DEPTH_8U, 1 );
		IplImage* temp_image = cvCreateImage(m_imageSize, IPL_DEPTH_8U, 1 );

		/* Preparation: This array will contain the features found in frame 1. */
		CvPoint2D32f frame1_features[400];
		/* Preparation: BEFORE the function call this variable is the array size
		 * (or the maximum number of features to find).  AFTER the function call
		 * this variable is the number of features actually found.
		 */
		int number_of_features;

		/* I'm hardcoding this at 400.  But you should make this a #define so that you can
		 * change the number of features you use for an accuracy/speed tradeoff analysis.
		 */
		number_of_features = 400;

		/* Actually run the Shi and Tomasi algorithm!!
		 * "frame1_1C" is the input image.
		 * "eig_image" and "temp_image" are just workspace for the algorithm.
		 * The first ".01" specifies the minimum quality of the features (based on the eigenvalues).
		 * The second ".01" specifies the minimum Euclidean distance between features.
		 * "NULL" means use the entire input image.  You could point to a part of the image.
		 * WHEN THE ALGORITHM RETURNS:
		 * "frame1_features" will contain the feature points.
		 * "number_of_features" will be set to a value <= 400 indicating the number of feature points found.
		 */
		cvGoodFeaturesToTrack(image1, eig_image, temp_image, frame1_features, &number_of_features, .03, .1, NULL);


		/* Pyramidal Lucas Kanade Optical Flow! */

		/* This array will contain the locations of the points from frame 1 in frame 2. */
		CvPoint2D32f frame2_features[400];
		/* The i-th element of this array will be non-zero if and only if the i-th feature of
		 * frame 1 was found in frame 2.
		 */
		char optical_flow_found_feature[400];
		/* The i-th element of this array is the error in the optical flow for the i-th feature
		 * of frame1 as found in frame 2.  If the i-th feature was not found (see the array above)
		 * I think the i-th entry in this array is undefined.
		 */
		float optical_flow_feature_error[400];

		/* This is the window size to use to avoid the aperture problem (see slide "Optical Flow: Overview"). */
		CvSize optical_flow_window = cvSize(15,15);
		
		/* This termination criteria tells the algorithm to stop when it has either done 20 iterations or when
		 * epsilon is better than .3.  You can play with these parameters for speed vs. accuracy but these values
		 * work pretty well in many situations.
		 */
		CvTermCriteria optical_flow_termination_criteria
			= cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 50, .9 );

		/* This is some workspace for the algorithm.
		 * (The algorithm actually carves the image into pyramids of different resolutions.)
		 */
		IplImage* pyramid1 = cvCreateImage( m_imageSize, IPL_DEPTH_8U, 1 );
		IplImage* pyramid2 = cvCreateImage( m_imageSize, IPL_DEPTH_8U, 1 );

		/* Actually run Pyramidal Lucas Kanade Optical Flow!!
		 * "frame1_1C" is the first frame with the known features.
		 * "frame2_1C" is the second frame where we want to find the first frame's features.
		 * "pyramid1" and "pyramid2" are workspace for the algorithm.
		 * "frame1_features" are the features from the first frame.
		 * "frame2_features" is the (outputted) locations of those features in the second frame.
		 * "number_of_features" is the number of features in the frame1_features array.
		 * "optical_flow_window" is the size of the window to use to avoid the aperture problem.
		 * "5" is the maximum number of pyramids to use.  0 would be just one level.
		 * "optical_flow_found_feature" is as described above (non-zero iff feature found by the flow).
		 * "optical_flow_feature_error" is as described above (error in the flow for this feature).
		 * "optical_flow_termination_criteria" is as described above (how long the algorithm should look).
		 * "0" means disable enhancements.  (For example, the second array isn't pre-initialized with guesses.)
		 */
		cvCalcOpticalFlowPyrLK(image1, image2, pyramid1, pyramid2, frame1_features, frame2_features, number_of_features, 
							   optical_flow_window, 5, optical_flow_found_feature, optical_flow_feature_error, 
							   optical_flow_termination_criteria, OPTFLOW_LK_GET_MIN_EIGENVALS);
			

		// get median of length and direction of all corresponding points
		vector<myLine>corresPoints;

		for(int i = 0; i < number_of_features; i++)
		{
			if ( optical_flow_found_feature[i] == 0 )	continue;

			Point2f a,b;
			a.x = (int) frame1_features[i].x;
			a.y = (int) frame1_features[i].y;
			b.x = (int) frame2_features[i].x;
			b.y = (int) frame2_features[i].y;

			corresPoints.push_back(myLine(a,b));
			
			double direction = atan2( (double) a.y - b.y, (double) b.x - b.x );
			directions.push_back(direction);
		}
		double sum_direction = std::accumulate(directions.begin(), directions.end(), 0.0);
		double mean_direction = sum_direction / directions.size();

		sort(corresPoints.begin(), corresPoints.end(),[](myLine a, myLine b) -> bool { return a.getLength() > b.getLength();});
		double median_lenght = corresPoints[(int)(corresPoints.size()/2)].getLength();
		
		sort(corresPoints.begin(), corresPoints.end(),[](myLine a, myLine b) -> bool { return a.getAngle() > b.getAngle();});
		double median_angle = corresPoints[(int)(corresPoints.size()/2)].getAngle();

		
		for(int i = 0; i < number_of_features; i++)
		{
			if ( optical_flow_found_feature[i] == 0 )	continue;

			Point2f a,b;
			a.x = (int) frame1_features[i].x;
			a.y = (int) frame1_features[i].y;
			b.x = (int) frame2_features[i].x;
			b.y = (int) frame2_features[i].y;

			double direction;		direction = atan2( (double) a.y - b.y, (double) b.x - b.x );
			double length;	length = sqrt( square(a.y - b.y) + square(a.x - b.x) );

			if (direction < mean_direction) {
				lengths1.push_back(length);		
			} else {
				lengths2.push_back(length);		
			}
		}

		double sum_length1 = std::accumulate(lengths1.begin(), lengths1.end(), 0.0);
		double sum_length2 = std::accumulate(lengths2.begin(), lengths2.end(), 0.0);
		double mean_length1 = sum_length1 / lengths1.size();
		double mean_length2 = sum_length2 / lengths2.size();


		//cout << "length1 : " << mean_length1 << "   ; length2  " << mean_length2 <<  "  : direction " << mean_direction << endl;


		// convert grayscale to color image
  		cvCvtColor(image1, colorImage, CV_GRAY2RGB);

		/* For fun (and debugging :)), let's draw the flow field. */
		for(int i = 0; i < number_of_features; i++)
		{
			/* If Pyramidal Lucas Kanade didn't really find the feature, skip it. */
			if ( optical_flow_found_feature[i] == 0 )	continue;

			int line_thickness;				line_thickness = 1;
			
			/* CV_RGB(red, green, blue) is the red, green, and blue components
			 * of the color you want, each out of 255.
			 */	
			CvScalar line_color;			
	
			/* Let's make the flow field look nice with arrows. */

			/* The arrows will be a bit too short for a nice visualization because of the high framerate
			 * (ie: there's not much motion between the frames).  So let's lengthen them by a factor of 3.
			 */
			Point2f p,q;
			p.x = (int) frame1_features[i].x;
			p.y = (int) frame1_features[i].y;
			q.x = (int) frame2_features[i].x;
			q.y = (int) frame2_features[i].y;

			double angle;		angle = atan2( (double) p.y - q.y, (double) p.x - q.x );
			double hypotenuse;	hypotenuse = sqrt( square(p.y - q.y) + square(p.x - q.x) );
			
			/* Here we lengthen the arrow by a factor of three. */
			q.x = (int) (p.x - 3 * hypotenuse * cos(angle));
			q.y = (int) (p.y - 3 * hypotenuse * sin(angle));

			if (angle < median_angle + 2 && angle > median_angle - 2 ) {
				if (hypotenuse < (median_lenght*3) && hypotenuse > 1.5 && hypotenuse > median_lenght*0.1) {
					drawLine(colorImage, p, q, angle, CV_RGB(255,0,0));
				} else {
					drawLine(colorImage, p, q, angle, CV_RGB(0,0,0));
					//cout << "1 do not draw " << hypotenuse << " angle: " << angle << ";   mean length: " << median_lenght << " ; mean_direction: " << median_angle << endl;
				}
			} else {
				//if (hypotenuse < (median_lenght*1.2) && hypotenuse > 1.5 && hypotenuse > median_lenght*0.8) {
				//	drawLine(colorImage, p, q, angle, CV_RGB(0,255,0));
				//} else {
					drawLine(colorImage, p, q, angle, CV_RGB(0,0,0));
					//cout << "2 do not draw " << hypotenuse << " angle: " << angle << ";    mean length: " << median_lenght << " ; mean_direction: " << median_angle << endl;
				//}
			}
		}
		

		/* Now display the image we drew on.  Recall that "Optical Flow" is the name of
		 * the window we created above.
		 */
		cvShowImage("Optical Flow", colorImage);

		// save image in every frame
		//string path = "data/image/current"+(to_string(frame))+".png";
		//cvSaveImage(path.c_str(), colorImage);

		// clear all data
		directions.clear();
		lengths1.clear();
		lengths2.clear();
		corresPoints.clear();
		cvReleaseImage(&eig_image);
		cvReleaseImage(&temp_image);
		cvReleaseImage(&pyramid1);
		cvReleaseImage(&pyramid2);
}


void drawLine (IplImage* ref, Point2f p, Point2f q, float angle, CvScalar const& color, int line_thickness ) {
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
