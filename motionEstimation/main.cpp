#include "MotionEstimation.h"
#include "Utility.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <vector>

// ************************************
// ******* Motion Estimation **********
// ************************************
// 1- Get Matrix K
// 2. calculate EssentialMatrix
// 3. for bundle adjustment use SSBA
// 4. or http://stackoverflow.com/questions/13921720/bundle-adjustment-functions
// 5. recover Pose (need newer version of calib3d)

//TODO:
// other meothod do decompose essential mat;
//      http://www.morethantechnical.com/2012/08/09/decomposing-the-essential-matrix-using-horn-and-eigen-wcode/
//

int main(){

    int frame=1;
    // get calibration Matrix K
    cv::Mat K_L, distCoeff_L, K_R, distCoeff_R;
    loadIntrinsic(K_L, K_R, distCoeff_L, distCoeff_R);

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

    cv::Mat P_0 = (cv::Mat_<double>(3,4) <<
                   1.0, 0.0, 0.0, 0.0,
                   0.0, 1.0, 0.0, 0.0,
                   0.0, 0.0, 1.0, 0.0 );

    cv::Mat R_0, T_0;
    decomposeProjectionMat(P_0, R_0, T_0);


    // define image size
    int resX = 752;
    int resY = 480;

    // currentPosition E Mat
    cv::Mat currentPos_ES_L = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat currentPos_ES_R = cv::Mat::eye(4, 4, CV_64F);

    // currentPosition SOLVE PNP RANSAC
    cv::Mat currentPos_PnP_L = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat currentPos_PnP_R = cv::Mat::eye(4, 4, CV_64F);

    // currentPosition TRIANGULATION
    cv::Mat currentPos_Stereo = cv::Mat::eye(4, 4, CV_64F);

    //load file names
    std::vector<string> filenames_left, filenames_right;
    getFiles("data/stereoImages/round-small/left/", filenames_left);
    getFiles("data/stereoImages/round-small/right/", filenames_right);

    initVisualisation();

    while (true){

        cout << "FRAME" <<  frame << endl;

        //stereo1
        cv::Mat image_L1 = cv::imread("data/stereoImages/round-small/left/"+filenames_left[frame],0);
        cv::Mat image_R1 = cv::imread("data/stereoImages/round-small/right/"+filenames_right[frame],0);

        //stereo2
        cv::Mat image_L2 = cv::imread("data/stereoImages/round-small/left/"+filenames_left[frame+1],0);
        cv::Mat image_R2 = cv::imread("data/stereoImages/round-small/right/"+filenames_right[frame+1],0);

        // Check for invalid input
        if(! image_L1.data || !image_R1.data || !image_R2.data || !image_L2.data) {
            cout <<  "Could not open or find the image: "  << std::endl ;
            //frame=1;
            continue;
        }

        // find Points ...
        std::vector<cv::Point2f> points_L1, points_R1, points_L2, points_R2;
        findCorresPoints_LucasKanade(image_L1, image_R1, image_L2, image_R2, &points_L1, &points_R1, &points_L2, &points_R2);

        // triangulate both stereo setups..
        // find inliers from median value
        std::vector<cv::Point2f> horizontal_L1, horizontal_R1, horizontal_L2, horizontal_R2;
        getInliersFromHorizontalDirection(make_pair(points_L1, points_R1), &horizontal_L1, &horizontal_R1);
        getInliersFromHorizontalDirection(make_pair(points_L2, points_R2), &horizontal_L2, &horizontal_R2);
        deleteZeroLines(horizontal_L1, horizontal_R1, horizontal_L2, horizontal_R2);

        std::vector<cv::Point2f> normP_L1, normP_R1, normP_L2, normP_R2;
        normalizePoints(KInv_L, KInv_R, horizontal_L1, horizontal_R1, normP_L1, normP_R1);
        normalizePoints(KInv_L, KInv_R, horizontal_L2, horizontal_R2, normP_L2, normP_R2);

        std::vector<cv::Point3f> pointCloud_1, pointCloud_2;
        TriangulatePointsHZ(P_0, P_LR, normP_L1, normP_R1, 0, pointCloud_1);
        TriangulatePointsHZ(P_0, P_LR, normP_L2, normP_R2, 0, pointCloud_2);

        // get RGB values for pointcloud representation
        std::vector<cv::Vec3b> RGBValues;
        for (unsigned int i = 0; i < horizontal_L1.size(); ++i){
            RGBValues.push_back(image_L1.at<cv::Vec3b>(horizontal_L1[i].x, horizontal_L1[i].y));
        }

        AddPointcloudToVisualizer(pointCloud_1, std::to_string(frame), RGBValues);


        // ######################## ESSENTIAL MAT ################################

        cv::Mat T_E_L, R_E_L, T_E_R, R_E_R;
        // UP TO SCALE!!!
        bool poseEstimationFoundES_L = motionEstimationEssentialMat(image_L1, image_L2, points_L1, points_L2, K_L, KInv_L, T_E_L, R_E_L);
        bool poseEstimationFoundES_R = motionEstimationEssentialMat(image_L1, image_L2, points_L1, points_L2, K_L, KInv_L, T_E_R, R_E_R);
        if (!poseEstimationFoundES_L){
            T_E_L = cv::Mat::zeros(3, 1, CV_64F);
            R_E_L = cv::Mat::eye(3, 3, CV_64F);
        }
        if (!poseEstimationFoundES_R){
            T_E_R = cv::Mat::zeros(3, 1, CV_64F);
            R_E_R = cv::Mat::eye(3, 3, CV_64F);
        }

        // find scale factors
        // find right scale factors u und v (according to rodehorst paper)
        // 1. method:
        double u_L1, u_R1;
        cv::Mat P_L, P_R;
        composeProjectionMat(T_E_L, R_E_L, P_L);
        composeProjectionMat(T_E_R, R_E_R, P_R);
        getScaleFactor(P_0, P_LR, P_L, P_R, normP_L1, normP_R1, normP_L2, normP_R2, u_L1, u_R1);
        cv::Mat T_E_L1 = T_E_L * u_L1;
        cv::Mat T_E_R1 = T_E_R * u_R1;

        // 2. method:
        double u_L2, u_R2;
        getScaleFactor2(T_LR, R_LR, T_E_L, R_E_L, T_E_R, u_L2, u_R2);
        cv::Mat T_E_L2 = T_E_L * u_L2;
        cv::Mat T_E_R2 = T_E_R * u_R2;

        //compare both methods
        //            cout << "u links  1: " << u_L1 << endl;
        //            cout << "u rechts 1: " << u_R1 << endl << endl;
        //            cout << "u links  2: " << u_L2 << endl;
        //            cout << "u rechts 2: " << u_R2 << endl;

        //LEFT:
        cv::Mat newPos_ES_L;
        getNewPos (currentPos_ES_L, T_E_L, R_E_L, newPos_ES_L);
        std::stringstream left_ES;
        left_ES << "camera_ES_left" << frame;
        addCameraToVisualizer(cv::Vec3f(T_E_L1), cv::Matx33f(R_E_L), 255, 0, 0, 20, left_ES.str());


        //RIGHT:
        cv::Mat newPos_ES_R;
        getNewPos (currentPos_ES_R, T_E_R, R_E_R, newPos_ES_R);
        std::stringstream right_ES;
        right_ES << "camera_ES_right" << frame;
        addCameraToVisualizer(cv::Vec3f(T_E_R1), cv::Matx33f(R_E_R), 125, 0, 0, 20, right_ES.str());

        currentPos_ES_L   = newPos_ES_L  ;
        currentPos_ES_R   = newPos_ES_R  ;
        // ##############################################################################




        // ################################## PnP ######################################
        cv::Mat T_PnP_L, R_PnP_L, T_PnP_R, R_PnP_R;
        bool poseEstimationFoundPnP_L = motionEstimationPnP(points_L2, pointCloud_1, K_L, T_PnP_L, R_PnP_L);
        bool poseEstimationFoundPnP_R = motionEstimationPnP(points_R2, pointCloud_1, K_R, T_PnP_R, R_PnP_R);
        if (!poseEstimationFoundPnP_L){
            T_PnP_L = cv::Mat::zeros(3, 1, CV_64F);
            R_PnP_L = cv::Mat::eye(3, 3, CV_64F);
        }
        if (!poseEstimationFoundPnP_R){
            T_PnP_R = cv::Mat::zeros(3, 1, CV_64F);
            R_PnP_R = cv::Mat::eye(3, 3, CV_64F);
        }

        //LEFT:
        cv::Mat newPos_PnP_L;
        getNewPos (currentPos_PnP_L, T_PnP_L, R_PnP_L, newPos_PnP_L);
        std::stringstream left_PnP;
        left_PnP << "camera_PnP_left" << frame;
        addCameraToVisualizer(cv::Vec3f(T_PnP_L), cv::Matx33f(R_PnP_L), 0, 255, 0, 20, left_PnP.str());


        //RIGHT:
        cv::Mat newPos_PnP_R;
        getNewPos (currentPos_ES_R, T_PnP_R, R_PnP_R, newPos_ES_R);
        std::stringstream right_PnP;
        right_PnP << "camera_PnP_right" << frame;
        addCameraToVisualizer(cv::Vec3f(T_PnP_R), cv::Matx33f(R_PnP_R), 0, 125, 0, 20, right_PnP.str());

        currentPos_PnP_L  = newPos_PnP_L ;
        currentPos_PnP_R  = newPos_PnP_R ;
        // ##############################################################################





        // ################################# STEREO #####################################

        cv::Mat T_Stereo, R_Stereo;
        bool poseEstimationFoundStereo = motionEstimationStereoCloudMatching(pointCloud_1, pointCloud_2, T_Stereo, R_Stereo);
        if (!poseEstimationFoundStereo){
            T_Stereo = cv::Mat::zeros(3, 1, CV_64F);
            R_Stereo = cv::Mat::eye(3, 3, CV_64F);
        }

        //STEREO:
        cv::Mat newPos_Stereo;
        getNewPos (currentPos_Stereo, T_Stereo, R_Stereo, newPos_Stereo);
        std::stringstream stereo;
        stereo << "camera_Stereo" << frame;
        addCameraToVisualizer(cv::Vec3f(T_Stereo), cv::Matx33f(R_Stereo), 0, 0, 255, 20, stereo.str());

        currentPos_Stereo = newPos_Stereo;
        // ##############################################################################


        RunVisualization();
        ++frame;

        // To Do:
        // swap image files...
    }
    return 0;
}