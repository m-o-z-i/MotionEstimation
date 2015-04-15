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
    int frame=0;

    //load file names
    std::vector<string> filenames_left, filenames_right;
    string dataPath = "data/maliksupertest/";
    getFiles(dataPath + "left/", filenames_left);
    getFiles(dataPath + "right/", filenames_right);

    // get calibration Matrix K
    cv::Mat K_L, distCoeff_L, K_R, distCoeff_R;
    loadIntrinsic(dataPath, K_L, K_R, distCoeff_L, distCoeff_R);

    // get extrinsic test parameter
    cv::Mat E_LR, F_LR, R_LR, T_LR;
    loadExtrinsic(dataPath, R_LR, T_LR, E_LR, F_LR);

    // load q matrix
    cv::Mat Q;
    cv::FileStorage fs(dataPath + "disparity/disparity_0.yml", cv::FileStorage::READ);
    fs["Q"] >> Q;
    fs.release();
    cout << Q << endl;

    //convert all to single precission
    K_L.convertTo(K_L, CV_32F);
    K_R.convertTo(K_R, CV_32F);
    distCoeff_L.convertTo(distCoeff_L, CV_32F);
    distCoeff_R.convertTo(distCoeff_R, CV_32F);
    E_LR.convertTo(E_LR, CV_32F);
    F_LR.convertTo(F_LR, CV_32F);
    R_LR.convertTo(R_LR, CV_32F);
    T_LR.convertTo(T_LR, CV_32F);
    Q.convertTo(Q, CV_32F);

    // calculate inverse K
    cv::Mat KInv_L, KInv_R;
    cv::invert(K_L, KInv_L);
    cv::invert(K_L, KInv_R);

    // get projection Mat between L and R
    cv::Mat P_LR, rvec_LR;
    composeProjectionMat(T_LR, R_LR, P_LR);
    cv::Rodrigues(R_LR, rvec_LR);

    cv::Mat P_0 = (cv::Mat_<float>(3,4) <<
                   1.0, 0.0, 0.0, 0.0,
                   0.0, 1.0, 0.0, 0.0,
                   0.0, 0.0, 1.0, 0.0 );

    cv::Mat R_0, T_0;
    decomposeProjectionMat(P_0, R_0, T_0);

    // define image size
    int resX = 752;
    int resY = 480;

    // currentPosition E Mat
    cv::Mat currentPos_ES_L = cv::Mat::eye(4, 4, CV_32F);
    cv::Mat currentPos_ES_R = cv::Mat::eye(4, 4, CV_32F);

    // currentPosition SOLVE PNP RANSAC
    cv::Mat currentPos_PnP_L = cv::Mat::eye(4, 4, CV_32F);
    cv::Mat currentPos_PnP_R = cv::Mat::eye(4, 4, CV_32F);

    // currentPosition TRIANGULATION
    cv::Mat currentPos_Stereo = cv::Mat::eye(4, 4, CV_32F);


    initVisualisation();

    while (true){

        cout << "FRAME" <<  frame << endl;

        //stereo1
        cv::Mat image_L1 = cv::imread(dataPath + "left/" + filenames_left[frame],0);
        cv::Mat image_R1 = cv::imread(dataPath + "right/"+ filenames_right[frame],0);

        //stereo2
        cv::Mat image_L2 = cv::imread(dataPath + "left/" + filenames_left[frame+1],0);
        cv::Mat image_R2 = cv::imread(dataPath + "right/"+ filenames_right[frame+1],0);

        // Check for invalid input
        if(! image_L1.data || !image_R1.data || !image_R2.data || !image_L2.data) {
            cout <<  "Could not open or find the image: "  << std::endl ;
            ++frame;
            continue;
        }

        // find Points ...
        std::vector<cv::Point2f> points_L1, points_R1, points_L2, points_R2;
        findCorresPoints_LucasKanade(image_L1, image_R1, image_L2, image_R2, points_L1, points_R1, points_L2, points_R2);

        //fastFeatureMatcher(image_L1, image_L2, image_L2, image_R2, points_L1, points_R1, points_L2, points_R2);

        std::vector<cv::Point2f> inlier_median_L1, inlier_median_R1, inlier_median_L2, inlier_median_R2;
        getInliersFromMedianValue(make_pair(points_L1, points_R1), inlier_median_L1, inlier_median_R1);
        getInliersFromMedianValue(make_pair(points_L2, points_R2), inlier_median_L2, inlier_median_R2);
        deleteZeroLines(inlier_median_L1, inlier_median_R1, inlier_median_L2, inlier_median_R2);


        std::vector<cv::Point2f> normP_L1, normP_R1, normP_L2, normP_R2;
        normalizePoints(KInv_L, KInv_R, points_L1, points_R1, normP_L1, normP_R1);
        normalizePoints(KInv_L, KInv_R, points_L2, points_R2, normP_L2, normP_R2);

        std::vector<cv::Point3f> pointCloud_1;
        TriangulatePointsHZ(P_0, P_LR, normP_L1, normP_R1, 0, pointCloud_1);


        // triangulate both stereo setups..
        // find inliers from median value
        std::vector<cv::Point2f> horizontal_L1, horizontal_R1, horizontal_L2, horizontal_R2;
        getInliersFromHorizontalDirection(make_pair(points_L1, points_R1), horizontal_L1, horizontal_R1);
        getInliersFromHorizontalDirection(make_pair(points_L2, points_R2), horizontal_L2, horizontal_R2);
        deleteZeroLines(horizontal_L1, horizontal_R1, horizontal_L2, horizontal_R2);

        if(0 == horizontal_L1.size()) {
            cout <<  "can't find any corresponding points in all 4 frames' "  << std::endl ;
            ++frame;
            continue;
        }

        std::vector<cv::Point2f> normP_L1_Trian, normP_R1_Trian, normP_L2_Trian, normP_R2_Trian;
        normalizePoints(KInv_L, KInv_R, horizontal_L1, horizontal_R1, normP_L1_Trian, normP_R1_Trian);
        normalizePoints(KInv_L, KInv_R, horizontal_L2, horizontal_R2, normP_L2_Trian, normP_R2_Trian);

        std::vector<cv::Point3f> pointCloud_inlier_1, pointCloud_inlier_2;
        std::vector<cv::Point2f> inlierTriang_L1, inlierTriang_R1, inlierTriang_L2, inlierTriang_R2;
        TriangulatePointsWithInlier(P_0, P_LR, normP_L1_Trian, normP_R1_Trian, 0, pointCloud_inlier_1, horizontal_L1, horizontal_R1, inlierTriang_L1, inlierTriang_R1);
        TriangulatePointsWithInlier(P_0, P_LR, normP_L2_Trian, normP_R2_Trian, 0, pointCloud_inlier_2, horizontal_L2, horizontal_R2, inlierTriang_L2, inlierTriang_R2);
        deleteZeroLines(inlierTriang_L1, inlierTriang_R1, inlierTriang_L2, inlierTriang_R2, pointCloud_inlier_1, pointCloud_inlier_2);


        // get RGB values for pointcloud representation
        std::vector<cv::Vec3b> RGBValues;
        for (unsigned int i = 0; i < horizontal_L1.size(); ++i){
            uchar grey = image_L1.at<uchar>(points_L1[i].x, points_L1[i].y);
            RGBValues.push_back(cv::Vec3b(0,255,0));
        }

        std::vector<cv::Vec3b> RGBValues2;
        for (unsigned int i = 0; i < horizontal_L1.size(); ++i){
            RGBValues2.push_back(cv::Vec3b(255,0,0));
        }

        //        rotatePointCloud(pointCloud_inlier_1);
        //        rotatePointCloud(pointCloud_inlier_2);

        //        rotatePointCloud(pointCloud_inlier_1, currentPos_Stereo);
        //        rotatePointCloud(pointCloud_inlier_2, currentPos_Stereo);

        int index = 0;
        for (auto i : pointCloud_inlier_1) {
            float length = sqrt( i.x*i.x + i.y*i.y + i.z*i.z);
            cout<< "HZ:  "<< index << ":  " << i << "   length: " << length << endl;
            ++index;
        }

        std::vector<cv::Point3f> pcloud_CV;
        TriangulateOpenCV(P_0, P_LR, K_L, K_R, inlierTriang_L1, inlierTriang_R1, pcloud_CV);

        index = 0;
        for (auto i : pcloud_CV) {
            float length = sqrt( i.x*i.x + i.y*i.y + i.z*i.z);
            cout<< "CV:  "<< index << ":  " << i << "   length: " << length << endl;
            ++index;
        }

        //AddPointcloudToVisualizer(pointCloud_inlier_1, "cloud1" + std::to_string(frame), RGBValues);
        //AddPointcloudToVisualizer(pointCloud_inlier_2, "cloud2" + std::to_string(frame), RGBValues2);

        //AddLineToVisualizer(pointCloud_inlier_1, pointCloud_inlier_2, "line"+std::to_string(frame), cv::Scalar(255,0,0));


#if 0
        // ######################## ESSENTIAL MAT ################################
        cv::Mat T_E_L, R_E_L, T_E_R, R_E_R;
        // UP TO SCALE!!!
        bool poseEstimationFoundES_L = motionEstimationEssentialMat(image_L1, points_L1, points_L2, K_L, KInv_L, T_E_L, R_E_L);
        bool poseEstimationFoundES_R = motionEstimationEssentialMat(image_R1, points_R1, points_R2, K_R, KInv_R, T_E_R, R_E_R);

        if (!poseEstimationFoundES_L){
            T_E_L = cv::Mat::zeros(3, 1, CV_32F);
            R_E_L = cv::Mat::eye(3, 3, CV_32F);
        }
        if (!poseEstimationFoundES_R){
            T_E_R = cv::Mat::zeros(3, 1, CV_32F);
            R_E_R = cv::Mat::eye(3, 3, CV_32F);
        }

        // find scale factors
        // find right scale factors u und v (according to rodehorst paper)
        // 1. method:
        float u_L1, u_R1;
        cv::Mat P_L, P_R;
        composeProjectionMat(T_E_L, R_E_L, P_L);
        composeProjectionMat(T_E_R, R_E_R, P_R);
        getScaleFactor(P_0, P_LR, P_L, P_R, normP_L1_Trian, normP_R1_Trian, normP_L2_Trian, normP_R2_Trian, u_L1, u_R1);
        cv::Mat T_E_L1 = T_E_L * u_L1;
        cv::Mat T_E_R1 = T_E_R * u_R1;

        // 2. method:
        float u_L2, u_R2;
        getScaleFactor2(T_LR, R_LR, T_E_L, R_E_L, T_E_R, u_L2, u_R2);
        cv::Mat T_E_L2 = T_E_L * u_L2;
        cv::Mat T_E_R2 = T_E_R * u_R2;

        //compare both methods
        cout << "u links  1: " << u_L1 << endl;
        cout << "u rechts 1: " << u_R1 << endl << endl;
        cout << "u links  2: " << u_L2 << endl;
        cout << "u rechts 2: " << u_R2 << endl;

        //LEFT:
        cv::Mat newPos_ES_L;
        getNewPos (currentPos_ES_L, T_E_L2, R_E_L, newPos_ES_L);
        std::stringstream left_ES;
        left_ES << "camera_ES_left" << frame;

        cv::Mat rotation_ES_L, translation_ES_L;
        decomposeProjectionMat(newPos_ES_L, translation_ES_L, rotation_ES_L);
        std::cout << "T_ES_left: " << translation_ES_L << std::endl;

        addCameraToVisualizer(translation_ES_L, rotation_ES_L, 255, 0, 0, 20, left_ES.str());


        //RIGHT:
        cv::Mat newPos_ES_R;
        getNewPos (currentPos_ES_R, T_E_R2, R_E_R, newPos_ES_R);
        std::stringstream right_ES;
        right_ES << "camera_ES_right" << frame;

        cv::Mat rotation_ES_R, translation_ES_R;
        decomposeProjectionMat(newPos_ES_R, translation_ES_R, rotation_ES_R);
        std::cout << "T_ES_right: " << translation_ES_R << std::endl;
        addCameraToVisualizer(translation_ES_R, rotation_ES_R, 0, 255, 0, 20, right_ES.str());

        currentPos_ES_L = newPos_ES_L;
        currentPos_ES_R = newPos_ES_R;
        // ##############################################################################
#endif

#if 0
        // ################################## PnP #######################################
        cv::Mat T_PnP_L, R_PnP_L, T_PnP_R, R_PnP_R;

        // GUESS TRANSLATION + ROTATION UP TO SCALE!!!
        bool poseEstimationFoundTemp_L = motionEstimationEssentialMat(image_L1, points_L1, points_L2, K_L, KInv_L, T_PnP_L, R_PnP_L);
        bool poseEstimationFoundTemp_R = motionEstimationEssentialMat(image_R1, points_R1, points_R2, K_R, KInv_R, T_PnP_R, R_PnP_R);

        if (!poseEstimationFoundTemp_L){
            T_PnP_L = cv::Mat::zeros(3, 1, CV_32F);
            R_PnP_L = cv::Mat::eye(3, 3, CV_32F);
        }
        if (!poseEstimationFoundTemp_R){
            T_PnP_R = cv::Mat::zeros(3, 1, CV_32F);
            R_PnP_R = cv::Mat::eye(3, 3, CV_32F);
        }

        // use initial guess values for pose estimation
        bool poseEstimationFoundPnP_L = motionEstimationPnP(points_L2, pointCloud_1, K_L, T_PnP_L, R_PnP_L);
        bool poseEstimationFoundPnP_R = motionEstimationPnP(points_R2, pointCloud_1, K_R, T_PnP_R, R_PnP_R);

        if (!poseEstimationFoundPnP_L){
            T_PnP_L = cv::Mat::zeros(3, 1, CV_32F);
            R_PnP_L = cv::Mat::eye(3, 3, CV_32F);
        }
        if (!poseEstimationFoundPnP_R){
            T_PnP_R = cv::Mat::zeros(3, 1, CV_32F);
            R_PnP_R = cv::Mat::eye(3, 3, CV_32F);
        }


        //LEFT:
        cv::Mat newPos_PnP_L;
        getNewPos (currentPos_PnP_L, T_PnP_L, R_PnP_L, newPos_PnP_L);
        std::stringstream left_PnP;
        left_PnP << "camera_PnP_left" << frame;

        cv::Mat rotation_PnP_L, translation_PnP_L;
        decomposeProjectionMat(newPos_PnP_L, translation_PnP_L, rotation_PnP_L);
        std::cout << "T_PnP_left: " << translation_PnP_L << std::endl;

        addCameraToVisualizer(translation_PnP_L, rotation_PnP_L, 0, 255, 0, 20, left_PnP.str());


        //RIGHT:
        cv::Mat newPos_PnP_R;
        getNewPos (currentPos_PnP_R, T_PnP_R, R_PnP_R, newPos_PnP_R);
        std::stringstream right_PnP;
        right_PnP << "camera_PnP_right" << frame;

        cv::Mat rotation_PnP_R, translation_PnP_R;
        decomposeProjectionMat(newPos_PnP_R, translation_PnP_R, rotation_PnP_R);
        std::cout << "T_PnP_left: " << translation_PnP_R << std::endl;

        addCameraToVisualizer(translation_PnP_R, rotation_PnP_R, 0, 125, 0, 20, right_PnP.str());


        currentPos_PnP_L  = newPos_PnP_L ;
        currentPos_PnP_R  = newPos_PnP_R ;
        // ##############################################################################
#endif

#if 1
        // ################################# STEREO #####################################
        // for cv::waitKey input:
//        drawPoints(image_L1, inlier_median_L1, "points links", cv::Scalar(255,0,0));
//        drawPoints(image_R1, inlier_median_R1, "points rechts", cv::Scalar(255,0,0));

        //load disparity map
        cv::Mat dispMap1;
        cv::FileStorage fs_dist1(dataPath + "disparity/disparity_"+to_string(frame)+".yml", cv::FileStorage::READ);
        fs_dist1["disparity"] >> dispMap1;
        fs_dist1.release();

        cv::Mat dispMap2;
        cv::FileStorage fs_dist2(dataPath + "disparity/disparity_"+to_string(frame+1)+".yml", cv::FileStorage::READ);
        fs_dist2["disparity"] >> dispMap2;
        fs_dist2.release();

        dispMap1.convertTo(dispMap1, CV_32F);
        dispMap2.convertTo(dispMap2, CV_32F);

        vector <cv::Mat_<float>> cloud1;
        vector <cv::Mat_<float>> cloud2;
        for(unsigned int i = 0; i < inlier_median_L1.size(); ++i){
            cv::Mat_<float> point3D1(1,4);
            cv::Mat_<float> point3D2(1,4);
            calcCoordinate(point3D1, Q, dispMap1, inlier_median_L1[i].x, inlier_median_L1[i].y);
            calcCoordinate(point3D2, Q, dispMap2, inlier_median_L2[i].x, inlier_median_L2[i].y);
            cloud1.push_back(point3D1);
            cloud2.push_back(point3D2);
        }

        std::vector<cv::Point3f> pcloud1, pcloud2;
        std::vector<cv::Vec3b> rgb1, rgb2;
        for (unsigned int i = 0; i < cloud1.size(); ++i) {
            if (!cloud1[i].empty() && !cloud2[i].empty()){
                pcloud1.push_back(cv::Point3f(cloud1[i](0), cloud1[i](1), cloud1[i](2) ));
                pcloud2.push_back(cv::Point3f(cloud2[i](0), cloud2[i](1), cloud2[i](2) ));
                rgb1.push_back(cv::Vec3b(255,0,0));
                rgb2.push_back(cv::Vec3b(0,255,0));
            }
        }

        AddPointcloudToVisualizer(pcloud1, "pcloud1", rgb1);
        AddPointcloudToVisualizer(pcloud2, "pcloud2", rgb2);






        cv::Mat T_Stereo, R_Stereo;
        bool poseEstimationFoundStereo = motionEstimationStereoCloudMatching(pcloud1, pcloud2, T_Stereo, R_Stereo);
        if (!poseEstimationFoundStereo){
            T_Stereo = cv::Mat::zeros(3, 1, CV_32F);
            R_Stereo = cv::Mat::eye(3, 3, CV_32F);
        }

        cout << "ROTATION \n" << endl;
        cout << R_Stereo << endl;
        cout << "\n TRANSLATION \n" << endl;
        cout << T_Stereo << endl;

        float x_angle, y_angle, z_angle;
        decomposeRotMat(R_Stereo, x_angle, y_angle, z_angle);
        cout << "x angle:"<< x_angle << endl;
        cout << "y angle:"<< y_angle << endl;
        cout << "z angle:"<< z_angle << endl;

        //STEREO:
        cv::Mat newPos_Stereo;
        getNewPos (currentPos_Stereo, T_Stereo, R_Stereo, newPos_Stereo);
        std::stringstream stereo;
        stereo << "camera_Stereo" << frame;

        cv::Mat rotation, translation;
        decomposeProjectionMat(newPos_Stereo, translation, rotation);
        //std::cout << "T: " << translation << std::endl;

        addCameraToVisualizer(translation, rotation, 0, 0, 255, 100, stereo.str());

        currentPos_Stereo = newPos_Stereo;
        // ##############################################################################
#endif

        RunVisualization();
        ++frame;
        cv::waitKey();

        // To Do:
        // swap image files...
    }
    return 0;
}
