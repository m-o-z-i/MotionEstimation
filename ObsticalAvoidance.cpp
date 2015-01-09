#include "bgapi2_genicam.hpp"
#include <opencv2/opencv.hpp>

#include <cmath>
#include <math.h> 
#include <vector>

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


CvSize 					m_imageSize(cvSize(1392, 1040));
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
void feature_tracking();

int main() {
	// initialize baumer camera
	init_camera();

	IplImage* image1 = cvCreateImage(m_imageSize, IPL_DEPTH_8U, 1);
	IplImage* image2 = cvCreateImage(m_imageSize, IPL_DEPTH_8U, 1);

	std::vector<double> lengths1;
	std::vector<double> lengths2;
	std::vector<double> directions;

	IplImage* image1_1C = cvCreateImage(m_imageSize, IPL_DEPTH_8U, 3);
	//IplImage* frame2_1C = cvCreateImage(m_imageSize, IPL_DEPTH_8U, 1);

	//cvConvertImage(image1, frame1_1C, CV_CVTIMG_FLIP);
	//cvConvertImage(image2, frame2_1C, CV_CVTIMG_FLIP);

	//cv::Mat* image = cv::Mat::zeros(m_imageSize, CV_8U);

	int frame=0;

	cvNamedWindow("Optical Flow", CV_WINDOW_AUTOSIZE);

	while(true)
	{

		++frame; 

		open_stream(image1);
		key = cvWaitKey(30);
		open_stream(image2);

		
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
		cvGoodFeaturesToTrack(image1, eig_image, temp_image, frame1_features, &number_of_features, .1, .1, NULL);


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
		CvSize optical_flow_window = cvSize(3,3);
		
		/* This termination criteria tells the algorithm to stop when it has either done 20 iterations or when
		 * epsilon is better than .3.  You can play with these parameters for speed vs. accuracy but these values
		 * work pretty well in many situations.
		 */
		CvTermCriteria optical_flow_termination_criteria
			= cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.3 );

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
		cvCalcOpticalFlowPyrLK(image1, image2, pyramid1, pyramid2, frame1_features, frame2_features, number_of_features, optical_flow_window, 5, optical_flow_found_feature, optical_flow_feature_error, optical_flow_termination_criteria, 0 );
			

		// get mean of length and direction of all corresponding points
		
		for(int i = 0; i < number_of_features; i++)
		{
			if ( optical_flow_found_feature[i] == 0 )	continue;

			CvPoint a,b;
			a.x = (int) frame1_features[i].x;
			a.y = (int) frame1_features[i].y;
			b.x = (int) frame2_features[i].x;
			b.y = (int) frame2_features[i].y;

			double direction;		direction = atan2( (double) a.y - b.y, (double) b.x - b.x );
			directions.push_back(direction);
		}
		double sum_direction = std::accumulate(directions.begin(), directions.end(), 0.0);
		double mean_direction = sum_direction / directions.size();


		for(int i = 0; i < number_of_features; i++)
		{
			if ( optical_flow_found_feature[i] == 0 )	continue;

			CvPoint a,b;
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


		cout << "length1 : " << mean_length1 << "   ; length2  " << mean_length2 <<  "  : direction " << mean_direction << endl;


		// convert grayscale to color image
  		cvCvtColor(image1, image1_1C, CV_GRAY2RGB);

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
			CvPoint p,q;
			p.x = (int) frame1_features[i].x;
			p.y = (int) frame1_features[i].y;
			q.x = (int) frame2_features[i].x;
			q.y = (int) frame2_features[i].y;

			double angle;		angle = atan2( (double) p.y - q.y, (double) p.x - q.x );
			double hypotenuse;	hypotenuse = sqrt( square(p.y - q.y) + square(p.x - q.x) );
			
			/* Here we lengthen the arrow by a factor of three. */
			q.x = (int) (p.x - 3 * hypotenuse * cos(angle));
			q.y = (int) (p.y - 3 * hypotenuse * sin(angle));

			if (hypotenuse > 0.1 && angle < mean_direction) {
				if (hypotenuse < (mean_length1*2)) {

					line_color = CV_RGB(255,0,0);

					/* Now we draw the main line of the arrow. */
					/* "frame1" is the frame to draw on.
					 * "p" is the point where the line begins.
					 * "q" is the point where the line stops.
					 * "CV_AA" means antialiased drawing.
					 * "0" means no fractional bits in the center cooridinate or radius.
					 */
					cvLine( image1_1C, p, q, line_color, line_thickness, CV_AA, 0 );
					/* Now draw the tips of the arrow.  I do some scaling so that the
					 * tips look proportional to the main line of the arrow.
					 */			
					p.x = (int) (q.x + 9 * cos(angle + pi / 4));
					p.y = (int) (q.y + 9 * sin(angle + pi / 4));
					cvLine( image1_1C, p, q, line_color, line_thickness, CV_AA, 0 );
					p.x = (int) (q.x + 9 * cos(angle - pi / 4));
					p.y = (int) (q.y + 9 * sin(angle - pi / 4));
					cvLine( image1_1C, p, q, line_color, line_thickness, CV_AA, 0 );
				} else {
					line_color = CV_RGB(0,0,0);

					/* Now we draw the main line of the arrow. */
					/* "frame1" is the frame to draw on.
					 * "p" is the point where the line begins.
					 * "q" is the point where the line stops.
					 * "CV_AA" means antialiased drawing.
					 * "0" means no fractional bits in the center cooridinate or radius.
					 */
					cvLine( image1_1C, p, q, line_color, line_thickness, CV_AA, 0 );
					/* Now draw the tips of the arrow.  I do some scaling so that the
					 * tips look proportional to the main line of the arrow.
					 */			
					p.x = (int) (q.x + 9 * cos(angle + pi / 4));
					p.y = (int) (q.y + 9 * sin(angle + pi / 4));
					cvLine( image1_1C, p, q, line_color, line_thickness, CV_AA, 0 );
					p.x = (int) (q.x + 9 * cos(angle - pi / 4));
					p.y = (int) (q.y + 9 * sin(angle - pi / 4));
					cvLine( image1_1C, p, q, line_color, line_thickness, CV_AA, 0 );
				}
			} else {
				if (hypotenuse < (mean_length2*2)) {

					line_color = CV_RGB(0,255,0);

					/* Now we draw the main line of the arrow. */
					/* "frame1" is the frame to draw on.
					 * "p" is the point where the line begins.
					 * "q" is the point where the line stops.
					 * "CV_AA" means antialiased drawing.
					 * "0" means no fractional bits in the center cooridinate or radius.
					 */
					cvLine( image1_1C, p, q, line_color, line_thickness, CV_AA, 0 );
					/* Now draw the tips of the arrow.  I do some scaling so that the
					 * tips look proportional to the main line of the arrow.
					 */			
					p.x = (int) (q.x + 9 * cos(angle + pi / 4));
					p.y = (int) (q.y + 9 * sin(angle + pi / 4));
					cvLine( image1_1C, p, q, line_color, line_thickness, CV_AA, 0 );
					p.x = (int) (q.x + 9 * cos(angle - pi / 4));
					p.y = (int) (q.y + 9 * sin(angle - pi / 4));
					cvLine( image1_1C, p, q, line_color, line_thickness, CV_AA, 0 );
				} else {
					line_color = CV_RGB(0,0,0);

					/* Now we draw the main line of the arrow. */
					/* "frame1" is the frame to draw on.
					 * "p" is the point where the line begins.
					 * "q" is the point where the line stops.
					 * "CV_AA" means antialiased drawing.
					 * "0" means no fractional bits in the center cooridinate or radius.
					 */
					cvLine( image1_1C, p, q, line_color, line_thickness, CV_AA, 0 );
					/* Now draw the tips of the arrow.  I do some scaling so that the
					 * tips look proportional to the main line of the arrow.
					 */			
					p.x = (int) (q.x + 9 * cos(angle + pi / 4));
					p.y = (int) (q.y + 9 * sin(angle + pi / 4));
					cvLine( image1_1C, p, q, line_color, line_thickness, CV_AA, 0 );
					p.x = (int) (q.x + 9 * cos(angle - pi / 4));
					p.y = (int) (q.y + 9 * sin(angle - pi / 4));
					cvLine( image1_1C, p, q, line_color, line_thickness, CV_AA, 0 );
				}
			}
		}
		

		/* Now display the image we drew on.  Recall that "Optical Flow" is the name of
		 * the window we created above.
		 */
		cvShowImage("Optical Flow", image1_1C);

		// save image in every frame
		string path = "data/image/current"+(to_string(frame))+".png";
		//cvSaveImage(path.c_str(), image1_1C);

		// clear all data
		directions.clear();
		lengths1.clear();
		lengths2.clear();
		cvReleaseImage(&eig_image);
		cvReleaseImage(&temp_image);
		cvReleaseImage(&pyramid1);
		cvReleaseImage(&pyramid2);
    	//cvReleaseImage(&calibrated_frame);

	}

	return 0;
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


	    //m_device->GetRemoteNode("Gain")->SetDouble(15.56);
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
