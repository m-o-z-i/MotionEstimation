#include "line/MyLine.h"
#include "FindCameraMatrices.h"
#include "FindPoints.h"
#include "MultiCameraPnP.h"
#include "Triangulation.h"
#include "Visualisation.h"

#include <cmath>
#include <math.h>
#include <vector>
#include <utility>

#include <sstream>
#include <string.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
//#include "opencv2/nonfree/nonfree.hpp"

using namespace std;

char key;




//TODO:
// other meothod do decompose essential mat;
//      http://www.morethantechnical.com/2012/08/09/decomposing-the-essential-matrix-using-horn-and-eigen-wcode/
//
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
 *
 * surf detection need nonfree lib... can't use it in the vr-lab
 */

int main() {
    int frame=1;
    while(true)
    {
        //stereo1
        cv::Mat frame1L = cv::imread("data/stereoImages/left/"+(std::to_string(frame))+"_l.jpg",0);
        cv::Mat frame1R = cv::imread("data/stereoImages/right/"+(std::to_string(frame))+"_r.jpg",0);

        //stereo2
        cv::Mat frame2L = cv::imread("data/stereoImages/left/"+(std::to_string(frame+1))+"_l.jpg",0);
        cv::Mat frame2R = cv::imread("data/stereoImages/right/"+(std::to_string(frame+1))+"_r.jpg",0);

        // Check for invalid input
        if(! frame1L.data || !frame1R.data || !frame2R.data || !frame2L.data)
        {
            cout <<  "Could not open or find the image: "  << std::endl ;
            frame=1;
            continue;
        }

        //drawAllStuff(mat_image11, mat_image12, mat_image21, mat_image22, frame);

        // ************************************
        // ******* Motion Estimation **********
        // ************************************
        // 1- Get Matrix K
        // 2. calculate EssentialMatrix
        // 3. for bundle adjustment use SSBA
        // 4. or http://stackoverflow.com/questions/13921720/bundle-adjustment-functions
        // 5. recover Pose (need newer version of calib3d)

        // find corresponding points
        vector<cv::Point2f> features = getStrongFeaturePoints(frame1L, 150,0.01,5);
        pair<vector<cv::Point2f>, vector<cv::Point2f>> corresPoints1to2 = refindFeaturePoints(frame1L, frame2L, features);
        pair<vector<cv::Point2f>, vector<cv::Point2f>> corresPointsLtoR = refindFeaturePoints(frame1L, frame1R, features);

        // compute fundemental matrix F
        vector<cv::Point2f> inliersF1, inliersF2;
        cv::Mat F;
        getFundamentalMatrix(corresPointsLtoR, &inliersF1, &inliersF2, F);

        vector<cv::Point2f> meanInliers1, meanInliers2;
        getInliersFromMeanValue(corresPointsLtoR, &meanInliers1, &meanInliers2);

        drawCorresPoints(frame1L, meanInliers1, meanInliers2, CV_RGB(255, 0, 0));

        // get calibration Matrix K
        cv::Mat K, distCoeff;
        cv::FileStorage fs("data/calibration/left.yml", cv::FileStorage::READ);
        fs["cameraMatrix"] >> K;
        fs["distCoeff"] >> distCoeff;
        fs.release();

        //load test essential mat
        cv::Mat ETest, FTest;
        cv::FileStorage fs2("data/calibration/extrinsic.yml", cv::FileStorage::READ);
          fs2["E"] >> ETest;
          fs2["F"] >> FTest;
          fs2.release();

        // get inverse K
        cv::Mat KInv;
        cv::invert(K, KInv);

        // calculate essential mat
        cv::Mat E = K.t() * F * K; //according to HZ (9.12)

//        std::cout << "EssentialMat \n" << E << std::endl;
//        std::cout << "\n\n EssentialMat TEST \n" << ETest << std::endl;
//        std::cout << "\n\n EssentialMat \n" << F << std::endl;
//        std::cout << "\n\n FundamentalMat Test\n" << FTest << std::endl;
//        cvWaitKey(0);

        std::vector<cv::Point2f> testPoints2D1 = {cv::Point2f(0.5,0.2), cv::Point2f(0.7,0.1), cv::Point2f(0.45,0.26), cv::Point2f(0.12,0.185)};
        std::vector<cv::Point2f> testPoints2D2 = {cv::Point2f(0.6,0.2), cv::Point2f(0.8,0.1), cv::Point2f(0.55,0.26), cv::Point2f(0.22,0.185)};

        // decompose right solution for R and T values and saved it to P1. get point cloud of triangulated points
        cv::Mat P1;
        std::vector<cv::Point3f> pointCloud;

        bool goodPFound = getRightProjectionMat(E, K, KInv, distCoeff, P1, testPoints2D1, testPoints2D2, pointCloud);

        if (goodPFound) {
//            std::cout << "#########################  " << frame  << "  ##############################" << std::endl;
//            std::cout << P1 << std::endl;
//            std::cout << "############################################################" << std::endl;

            cv::Mat KNew, RNew, TNew;
            cv::decomposeProjectionMatrix(P1, KNew, RNew, TNew);
            double n = TNew.at<double>(3,0);
            double x = TNew.at<double>(0,0)/n;
            double y = TNew.at<double>(1,0)/n;
            double z = TNew.at<double>(2,0)/n;

            // cout << "cameraPos: [" << x << ", " << y << ", " << z << "]"<< endl;
        } else {
            // cout << "no motion found" << endl;
        }

        ++frame;
        cvWaitKey(0);
    }
    return 0;
}
























