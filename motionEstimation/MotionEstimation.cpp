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
    int frame=17;

    // get calibration Matrix K
    cv::Mat KL, distCoeffL, KR, distCoeffR;
    loadIntrinsic("left", KL, distCoeffL);
    loadIntrinsic("right", KR, distCoeffR);

    // get extrinsic test parameter
    cv::Mat ETest, FTest, RTest, TTest;
    loadExtrinsic(RTest, TTest, ETest, FTest);

    // calculate inverse K
    cv::Mat KLInv, KRInv;
    cv::invert(KL, KLInv);
    cv::invert(KL, KRInv);

    // define image size
    int resX = 752;
    int resY = 480;

    while(true)
    {
        // ************************************
        // ******* Motion Estimation **********
        // ************************************
        // 1- Get Matrix K
        // 2. calculate EssentialMatrix
        // 3. for bundle adjustment use SSBA
        // 4. or http://stackoverflow.com/questions/13921720/bundle-adjustment-functions
        // 5. recover Pose (need newer version of calib3d)

        //stereo1
        cv::Mat frame1L = cv::imread("data/stereoImages/left/"+(std::to_string(frame))+"_l.jpg",0);
        cv::Mat frame1R = cv::imread("data/stereoImages/right/"+(std::to_string(frame))+"_r.jpg",0);

        //stereo2
        cv::Mat frame2L = cv::imread("data/stereoImages/left/"+(std::to_string(frame+1))+"_l.jpg",0);
        cv::Mat frame2R = cv::imread("data/stereoImages/right/"+(std::to_string(frame+1))+"_r.jpg",0);

        // Check for invalid input
        if(! frame1L.data || !frame1R.data || !frame2R.data || !frame2L.data) {
            cout <<  "Could not open or find the image: "  << std::endl ;
            frame=1;
            continue;
        }

        // find corresponding points
        vector<cv::Point2f> features = getStrongFeaturePoints(frame1L, 10,0.01,5);
        pair<vector<cv::Point2f>, vector<cv::Point2f>> corresPoints1to2 = refindFeaturePoints(frame1L, frame2L, features);
        pair<vector<cv::Point2f>, vector<cv::Point2f>> corresPointsL1toR1 = refindFeaturePoints(frame1L, frame1R, corresPoints1to2.first);
        pair<vector<cv::Point2f>, vector<cv::Point2f>> corresPointsL2toR2 = refindFeaturePoints(frame1L, frame1R, corresPoints1to2.second);

        // delete in all frames points, that are not visible in each frames
        deleteUnvisiblePoints(corresPoints1to2, corresPointsL1toR1, corresPointsL2toR2, resX, resY);

        // find inliers with median value
        vector<cv::Point2f> inliersMedianL1, inliersMedianR1;
        getInliersFromMedianValue(make_pair(corresPointsL1toR1.first, corresPointsL1toR1.second), &inliersMedianL1, &inliersMedianR1);
        // make sure that there are this inlier in all frames. If not delete this inlier in all frames
        deleteZeroLines(inliersMedianL1, inliersMedianR1);

        // compute fundemental matrix F
        cv::Mat F;
        vector<cv::Point2f> inliersFL1, inliersFR1;
        getFundamentalMatrix(make_pair(inliersMedianL1, inliersMedianR1), &inliersFL1, &inliersFR1, F);

        // make sure that there are this inlier in all frames. If not delete this inlier in all frames
        deleteZeroLines(inliersFL1, inliersFR1);

        //visualisize
        drawCorresPoints(frame1L, corresPointsL1toR1.first, corresPointsL1toR1.second, "Found CorresPoints", CV_RGB(0,255,0));
        drawCorresPoints(frame1L, inliersMedianL1, inliersMedianR1, "Inliers Median", CV_RGB(255,255,0));
        drawCorresPoints(frame1L, inliersFL1, inliersFR1, "inliers after ransac. for F computation", CV_RGB(0,255,255));
        drawEpipolarLines(frame1L, frame1R, inliersFL1, inliersFR1, F);

        // normalisize all Points

        // calculate essential mat
        cv::Mat E = KR.t() * F * KL; //according to HZ (9.12)

//        std::cout << "\n\n FundamentalMat \n" << F << std::endl;
//        std::cout << "\n\n FundamentalMat Test\n" << FTest << std::endl;
//        std::cout << "EssentialMat \n" << E << std::endl;
//        std::cout << "\n\n EssentialMat TEST \n" << ETest << std::endl;
//        cvWaitKey(0);

        // decompose right solution for R and T values and saved it to P1. get point cloud of triangulated points
        cv::Mat P1;
        std::vector<cv::Point3f> pointCloud;
        bool goodPFound = getRightProjectionMat(E, KL, KLInv, distCoeffL, P1, inliersFL1, inliersFR1, pointCloud);

        if (goodPFound) {
            cv::Mat KNew, RNew, TNew, RotX, RotY, RotZ, EulerRot;
            cv::decomposeProjectionMatrix(P1, KNew, RNew, TNew, RotX, RotY, RotZ, EulerRot);
            double n = TNew.at<double>(3,0);
            double x = TNew.at<double>(0,0)/n;
            double y = TNew.at<double>(1,0)/n;
            double z = TNew.at<double>(2,0)/n;

            //cout << "cameraPos: [" << x << ", " << y << ", " << z << "]  "<< endl <<" rotation: " << endl << EulerRot << endl;
        } else {
            // cout << "no motion found" << endl;
        }

        // cv::Mat_<double> rvec, t, R;
        // findPoseEstimation(rvec,t,R,pointCloud,medianInliersR1, K, distCoeff);

        ++frame;
        cvWaitKey(0);
    }
    return 0;
}
























