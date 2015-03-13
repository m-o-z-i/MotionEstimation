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
#include <stack>
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
    // get calibration Matrix K
    cv::Mat K_L, distCoeff_L, K_R, distCoeff_R;
    loadIntrinsic("left", K_L, distCoeff_L);
    loadIntrinsic("right", K_R, distCoeff_R);

    // get extrinsic test parameter
    cv::Mat E_LR, F_LR, R_LR, T_LR;
    loadExtrinsic(R_LR, T_LR, E_LR, F_LR);

    // calculate inverse K
    cv::Mat KInv_L, KInv_R;
    cv::invert(K_L, KInv_L);
    cv::invert(K_L, KInv_R);

    // get projection Mat between L and R
    cv::Mat P_LR;
    cv::hconcat(R_LR, T_LR, P_LR);

    // define image size
    int resX = 752;
    int resY = 480;

    // currentPosition
    cv::Point2f currentPos_L1(512, 512);
    cv::Point2f currentPos_R1(512, 512);
    cv::Point2f currentPos_L2(512, 512);
    cv::Point2f currentPos_R2(512, 512);

    cv::Mat path1 = cv::imread("data/background.jpg");
    cv::Mat path2 = cv::imread("data/background.jpg");

    cv::namedWindow("motionPath 1", cv::WINDOW_NORMAL);
    cv::namedWindow("motionPath 2", cv::WINDOW_NORMAL);

    while(true)
    {
        cout << "FRAME" <<  frame << endl;
        // ************************************
        // ******* Motion Estimation **********
        // ************************************
        // 1- Get Matrix K
        // 2. calculate EssentialMatrix
        // 3. for bundle adjustment use SSBA
        // 4. or http://stackoverflow.com/questions/13921720/bundle-adjustment-functions
        // 5. recover Pose (need newer version of calib3d)

        //stereo1
        cv::Mat frame_L1 = cv::imread("data/stereoImages/left/"+(std::to_string(frame))+"_l.jpg",0);
        cv::Mat frame_R1 = cv::imread("data/stereoImages/right/"+(std::to_string(frame))+"_r.jpg",0);

        //stereo2
        cv::Mat frame_L2 = cv::imread("data/stereoImages/left/"+(std::to_string(frame+1))+"_l.jpg",0);
        cv::Mat frame_R2 = cv::imread("data/stereoImages/right/"+(std::to_string(frame+1))+"_r.jpg",0);

        // Check for invalid input
        if(! frame_L1.data || !frame_R1.data || !frame_R2.data || !frame_L2.data) {
            cout <<  "Could not open or find the image: "  << std::endl ;
            //frame=1;
            continue;
        }

        // find corresponding points
        vector<cv::Point2f> features = getStrongFeaturePoints(frame_L1, 250,0.01,5);
        pair<vector<cv::Point2f>, vector<cv::Point2f>> corresPointsL1toR1 = refindFeaturePoints(frame_L1, frame_R1, features);
        pair<vector<cv::Point2f>, vector<cv::Point2f>> corresPointsL1toL2 = refindFeaturePoints(frame_L1, frame_L2, corresPointsL1toR1.first);
        pair<vector<cv::Point2f>, vector<cv::Point2f>> corresPointsR1toR2 = refindFeaturePoints(frame_R1, frame_R2, corresPointsL1toR1.second);

        // delete in all frames points, that are not visible in each frames
        deleteUnvisiblePoints(corresPointsL1toR1, corresPointsL1toL2, corresPointsR1toR2, resX, resY);


        // find inliers from median value
        vector<cv::Point2f> inliersMedian_L1a, inliersMedian_R1a;
        getInliersFromMedianValue(make_pair(corresPointsL1toR1.first, corresPointsL1toR1.second), &inliersMedian_L1a, &inliersMedian_R1a);

        vector<cv::Point2f> inliersMedian_L1b, inliersMedian_L2;
        getInliersFromMedianValue(make_pair(corresPointsL1toL2.first, corresPointsL1toL2.second), &inliersMedian_L1b, &inliersMedian_L2);

        vector<cv::Point2f> inliersMedian_R1b, inliersMedian_R2;
        getInliersFromMedianValue(make_pair(corresPointsR1toR2.first, corresPointsR1toR2.second), &inliersMedian_R1b, &inliersMedian_R2);

        deleteZeroLines(inliersMedian_L1a, inliersMedian_R1a, inliersMedian_L1b, inliersMedian_L2, inliersMedian_R1b, inliersMedian_R2);

        if (8 > inliersMedian_R1a.size()) {
            cout << "to less points found" << endl;
            ++frame;
            continue;
        }

        // compute fundemental matrix FL1L2
        cv::Mat F_L;
        bool foundF_L;
        vector<cv::Point2f> inliersF_L1, inliersF_L2;
        foundF_L = getFundamentalMatrix(make_pair(inliersMedian_L1a, inliersMedian_L2), &inliersF_L1, &inliersF_L2, F_L);

        // compute fundemental matrix FL1L2
        cv::Mat F_R;
        bool foundF_R;
        vector<cv::Point2f> inliersF_R1, inliersF_R2;
        foundF_R = getFundamentalMatrix(make_pair(inliersMedian_R1a, inliersMedian_R2), &inliersF_R1, &inliersF_R2, F_R);

        // can't find fundamental Mat
        if (!foundF_L || !foundF_R){
            cout << "can't find F" << endl;
            ++frame;
            continue;
        }

        // make sure that there are this inlier in all frames. If not delete this inlier in all frames
        deleteZeroLines(inliersF_L1, inliersF_L2, inliersF_R1, inliersF_R2);

        //visualisize
        // convert grayscale to color image
//        cv::Mat color_image;
//        cv::cvtColor(frame1L, color_image, CV_GRAY2RGB);
//        drawCorresPoints(color_image, inliersMedianL1a, inliersMedianR1a, "Found CorresPoints l1a r1a", CV_RGB(0,255,0));
//        drawCorresPoints(color_image, inliersMedianL1b, inliersMedianR1b, "Found CorresPoints l1b r1b", CV_RGB(0,255,0));

//        drawCorresPoints(color_image, inliersMedianL1a, inliersMedianL2, "Found CorresPoints L1 To L2", CV_RGB(255,0,0));
//        drawCorresPoints(color_image, inliersMedianR1b, inliersMedianR2, "Found CorresPoints R1 To R2", CV_RGB(0,0,255));
//        drawCorresPoints(color_image, corresPoints1to2.first, corresPoints1to2.second, "Found CorresPoints", CV_RGB(0,255,0));
//        drawCorresPoints(color_image, corresPointsL1toR1.first, corresPointsL1toR1.second, "Found CorresPoints", CV_RGB(255,0,0));
//        drawCorresPoints(color_image, corresPointsL1toR2.first, corresPointsL1toR2.second, "Found CorresPoints", CV_RGB(0,0,255));

//        drawCorresPoints(color_image, inliersMedianL1, inliersMedianR1, "Inliers Median", CV_RGB(255,255,0));
//        drawCorresPoints(color_image, inliersFL1, inliersFR1, "inliers after ransac. for F computation", CV_RGB(0,255,255));
//        drawEpipolarLines(frame1L, frame1R, inliersFL1, inliersFR1, F);

        // normalisize all Points
        vector<cv::Point2f> normPoints_L1, normPoints_R1, normPoints_L2, normPoints_R2;
        normalizePoints(KInv_L, KInv_R, inliersF_L1, inliersF_R1, normPoints_L1, normPoints_R1);
        normalizePoints(KInv_L, KInv_R, inliersF_L2, inliersF_R2, normPoints_L2, normPoints_R2);

        // calculate essential mat
        cv::Mat E_L = K_R.t() * F_L * K_L; //according to HZ (9.12)
        cv::Mat E_R = K_R.t() * F_R * K_L; //according to HZ (9.12)

//        std::cout << "\n\n FundamentalMat \n" << F << std::endl;
//        std::cout << "\n\n FundamentalMat Test\n" << FTest << std::endl;
//        std::cout << "EssentialMat \n" << E << std::endl;
//        std::cout << "\n\n EssentialMat TEST \n" << ETest << std::endl;
//        cvWaitKey(0);

        // decompose right solution for R and T values and saved it to P1. get point cloud of triangulated points
        cv::Mat P_L, P_R;
        std::vector<cv::Point3f> pointCloud_L, pointCloud_R;
        bool goodPFound_L = getRightProjectionMat(E_L, P_L, normPoints_L1, normPoints_L2, pointCloud_L);
        bool goodPFound_R = getRightProjectionMat(E_R, P_R, normPoints_R1, normPoints_R2, pointCloud_R);

        if (goodPFound_L && goodPFound_R) {
            // find right scale factors u und v (according to rodehorst paper)
            // 1. method:
            cv::Mat P0 = (cv::Mat_<double>(3,4) <<
                          1.0, 0.0, 0.0, 0.0,
                          0.0, 1.0, 0.0, 0.0,
                          0.0, 0.0, 1.0, 0.0 );

            std::vector<cv::Point3f> X, X_L, X_R;
            TriangulatePointsHZ(P0, P_LR, normPoints_L1, normPoints_R1, 5, X);
            TriangulatePointsHZ(P0, P_L , normPoints_L1, normPoints_L2, 5, X_L);
            TriangulatePointsHZ(P0, P_R , normPoints_R1, normPoints_R2, 5, X_R);

            double sum_L = 0;
            double sum_R = 0;
            for (unsigned int i = 0; i < X.size(); ++i) {
                sum_L += ((cv::norm(X[i])*1.0) / cv::norm(X_L[i])*1.0);
                sum_R += ((cv::norm(X[i])*1.0) / cv::norm(X_R[i])*1.0);
            }

            double u_L = 1.0/X.size() * sum_L;
            double u_R = 1.0/X.size() * sum_R;

            // 2. method:
            cv::Mat R_L, T_L, R_R, T_R;
            decomposeProjectionMat(P_L, R_L, T_L);
            decomposeProjectionMat(P_R, R_R, T_R);

            cv::Mat A(3, 2, CV_32F);
            cv::Mat B(3, 1, CV_32F);
            cv::Mat x(2, 1, CV_32F);

            cv::hconcat(T_L, -(R_LR*T_R), A);
            B = T_LR -(R_L*T_LR);

//            cout << "\n\n\n  ########### Matrizen ############### \n A: \n "<< A << endl << endl;
//            cout << "\n B: \n "<< B << endl << endl;
//            cout << "\n x: \n "<< x << endl << endl;

            //solve Ax = B
            cv::solve(A, B, x, cv::DECOMP_SVD);
            double u_L2 = x.at<double>(0,0);
            double u_R2 = x.at<double>(1,0);

            // compare both methods
//            cout << "u links  1: " << u_L << endl;
//            cout << "u rechts 1: " << u_R << endl << endl;
//            cout << "u links  2: " << u_L2 << endl;
//            cout << "u rechts 2: " << u_R2 << endl;

            //visualisize
            currentPos_L1 = drawCameraPath(path1, currentPos_L1, T_L * (u_L/100.0), "motionPath 1", cv::Scalar(255,0,0));
            currentPos_R1 = drawCameraPath(path1, currentPos_R1, T_R * (u_R/100.0), "motionPath 1", cv::Scalar(255,255,0));
            currentPos_L2 = drawCameraPath(path2, currentPos_L2, T_L * u_L2, "motionPath 2", cv::Scalar(255,0,0));
            currentPos_R2 = drawCameraPath(path2, currentPos_R2, T_R * u_R2, "motionPath 2", cv::Scalar(0,255,0));


//            cv::Mat KNew, RNew, TNew, RotX, RotY, RotZ, EulerRot;
//            cv::decomposeProjectionMatrix(P_L, KNew, RNew, TNew, RotX, RotY, RotZ, EulerRot);
//            cout << P_L << endl << RNew << endl << TNew << endl;

//            double n = TNew.at<double>(3,0);
//            double x = TNew.at<double>(0,0)/n;
//            double y = TNew.at<double>(1,0)/n;
//            double z = TNew.at<double>(2,0)/n;

//            cv::Vec3f TVec(x, y, z);
//            cv::Vec3f TVecTest(T_LR);

//            double length1 = sqrt(TVec[0] * TVec[0] + TVec[1] * TVec[1] + TVec[2] *TVec[2] );
//            double length2 = sqrt(TVecTest[0] * TVecTest[0] + TVecTest[1] * TVecTest[1] + TVecTest[2] *TVecTest[2] );

//            cout << "cameraRot: owndata " << endl << EulerRot << endl;
//            cout << "cameraPos: owndata [" << TVec[0]/length1 << ", " << TVec[1]/length1 << ", " << TVec[2]/length1 << "]"  << endl;
//            cout << "cameraPos: hagen   [" << TVecTest[0]/length2 << ", " << TVecTest[1]/length2 << ", " << TVecTest[2]/length2 << "]"   << endl;
//            cout << "cameraRot: hagen " << endl << RTest << endl;

        } else {
            cout << "can't estimate motion no perspective Mat Found" << endl;
        }

        // cv::Mat_<double> rvec, t, R;
        // findPoseEstimation(rvec,t,R,pointCloud,medianInliersR1, K, distCoeff);

        ++frame;
        cvWaitKey(10);
    }
    cvWaitKey(0);
    return 0;
}
























