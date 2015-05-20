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

    //load config file
    int mode = 0;
    string dataPath;
    cv::FileStorage config("data/config.yml", cv::FileStorage::READ);
    config["mode"] >> mode;
    config["path"] >> dataPath;
    config.release();

    //load file names
    std::vector<string> filenames_left, filenames_right;
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

    // currentPosition E Mat
    cv::Mat currentPos_ES_L = cv::Mat::eye(4, 4, CV_32F);
    cv::Mat currentPos_ES_R = cv::Mat::eye(4, 4, CV_32F);

    // currentPosition SOLVE PNP RANSAC
    cv::Mat currentPos_PnP_L = cv::Mat::eye(4, 4, CV_32F);
    cv::Mat currentPos_PnP_R = cv::Mat::eye(4, 4, CV_32F);

    // currentPosition TRIANGULATION
    cv::Mat currentPos_Stereo = cv::Mat::eye(4, 4, CV_32F);

    initVisualisation();

    // key input
    // stop and play with space and with n go to next frame
    char key = 0;
    bool loop = true;

    int frame1 = 0;
    int frame2 = frame1;

    bool skipFrame = true;
    int skipFrameNumber = 0;

    while (true){
        frame1 = frame2;

        // load stereo1
        cv::Mat image_L1 = cv::imread(dataPath + "left/" + filenames_left[frame1],0);
        cv::Mat image_R1 = cv::imread(dataPath + "right/"+ filenames_right[frame1],0);

        // Check for invalid input
        if(! image_L1.data || !image_R1.data) {
            cout <<  "Could not open or find the image from stereo 1: "  << std::endl ;
            ++frame1;
            frame2 = frame1;
            continue;
        }

        // find points in frame 1 ..
        std::vector<cv::Point2f> features = getStrongFeaturePoints(image_L1, 100, 0.001, 20);
        std::vector<cv::Point2f> points_L1_temp, points_R1_temp;
        refindFeaturePoints(image_L1, image_R1, features, points_L1_temp, points_R1_temp);

        // skip frame if no features are found in both images
        if (10 > points_L1_temp.size()) {
            cout <<  "Could not find more than features in stereo 1: "  << std::endl ;
            ++frame1;
            frame2 = frame1;
            continue;
        }

        skipFrameNumber = 0;
        skipFrame = true;

        while (skipFrame){
            ++skipFrameNumber;
            skipFrame = false;

            // skip no more than 4 frames
            if(1 < skipFrameNumber){
                frame1 = frame2;
                std::cout << "################### NO MOVEMENT FOR LAST 4 FRAMES ####################" << std::endl;
                break;
            }
            ++frame2;

            cout << "\n\n########################## FRAME "<<  frame1 << "  zu   " << frame2 << " ###################################" << endl;

            // load stereo2
            cv::Mat image_L2 = cv::imread(dataPath + "left/" + filenames_left[frame2],0);
            cv::Mat image_R2 = cv::imread(dataPath + "right/"+ filenames_right[frame2],0);

            // Check for invalid input
            if(! image_L2.data || !image_R2.data) {
                cout <<  "Could not open or find the image from stereo 2: "  << std::endl ;
                skipFrame = true;
                continue;
            }

            // find stereo 1 points in stereo 2 ...
            std::vector<cv::Point2f> points_L1, points_R1, points_L2, points_R2;
            refindFeaturePoints(image_L1, image_L2, points_L1_temp, points_L1, points_L2);
            refindFeaturePoints(image_R1, image_R2, points_R1_temp, points_R1, points_R2);
            // delete in all frames points, that are not visible in each frames
            deleteUnvisiblePoints(points_L1_temp, points_R1_temp, points_L1, points_R1, points_L2, points_R2, image_L1.cols, image_L1.rows);
            //fastFeatureMatcher(image_L1, image_L2, image_L2, image_R2, points_L1, points_R1, points_L2, points_R2);

            // skip frame if no features are found in both images
            if (0 == points_L1.size()) {
                cout <<  "Could not find features in stereo 2: "  << std::endl ;
                skipFrame = true;
                continue;
            }

            if (1 == mode) {
                // ######################## ESSENTIAL MAT ################################
                // compute F and get inliers from Ransac

                // skip frames if there are too less points found
                if (8 > points_L1.size()) {
                    cout << "NO MOVEMENT: to less points found" << endl;
                    skipFrame = true;
                    continue;
                }


//                // convert grayscale to color image and draw all points
//                cv::Mat color_image;
//                cv::cvtColor(image_L1, color_image, CV_GRAY2RGB);

                //drawCorresPointsRef(color_image, points_L1, points_L2, "all points left", cv::Scalar(255,0,0));

                // get inlier from stereo constraints
                std::vector<cv::Point2f> inliersHorizontal_L1, inliersHorizontal_R1, inliersHorizontal_L2, inliersHorizontal_R2;
                getInliersFromHorizontalDirection(make_pair(points_L1, points_R1), inliersHorizontal_L1, inliersHorizontal_R1);
                getInliersFromHorizontalDirection(make_pair(points_L2, points_R2), inliersHorizontal_L2, inliersHorizontal_R2);
                //delete all points that are not correctly found in stereo setup
                deleteZeroLines(points_L1, points_R1, points_L2, points_R2, inliersHorizontal_L1, inliersHorizontal_R1, inliersHorizontal_L2, inliersHorizontal_R2);

                // skip frame because something fails with rectification (ex. frame 287 dbl)
                if (8 > inliersHorizontal_L1.size()) {
                    cout << "NO MOVEMENT: couldn't find horizontal points... probably rectification fails or to less feature points found?!" << endl;
                    skipFrame = true;
                    continue;
                }


                // compute fundemental matrix F_L1L2
                cv::Mat F_L;
                bool foundF_L;
                std::vector<cv::Point2f> inliersF_L1, inliersF_L2;
                foundF_L = getFundamentalMatrix(points_L1, points_L2, &inliersF_L1, &inliersF_L2, F_L);

                // compute fundemental matrix F_R1R2
                cv::Mat F_R;
                bool foundF_R;
                std::vector<cv::Point2f> inliersF_R1, inliersF_R2;
                foundF_R = getFundamentalMatrix(points_R1, points_R2, &inliersF_R1, &inliersF_R2, F_R);

                // make sure that there are all inliers in all frames.
                deleteZeroLines(inliersF_L1, inliersF_L2, inliersF_R1, inliersF_R2);


                // skip frame because something fails with rectification (ex. frame 287 dbl)
                // TODO: check how often this happens
                if (1 > inliersF_L1.size()) {
                    cout << "NO MOVEMENT: couldn't find enough ransac inlier" << endl;
                    skipFrame = true;
                    continue;
                }

                drawCorresPoints(image_L1, inliersF_L1, inliersF_L2, "inlier F left " , CV_RGB(0,0,255));
                drawCorresPoints(image_R1, inliersF_R1, inliersF_R2, "inlier F right " , CV_RGB(0,0,255));

//                // draw inliers
//                drawCorresPointsRef(color_image,inliersHorizontal_L1,  inliersHorizontal_L2, "inlier horizontal left", cv::Scalar(0,0,255));
//                drawCorresPointsRef(color_image, inliersF_L1, inliersF_L2, "inlier points left", cv::Scalar(0,255,0));

//                char key2 = cv::waitKey();
//                if (char(key2) == 's'){
//                    cv::imwrite("data/docu/inlier_outlier.jpg", color_image);
//                }

                cv::Mat T_E_L, R_E_L, T_E_R, R_E_R;
                // UP TO SCALE!!!
                bool poseEstimationFoundES_L = false;
                bool poseEstimationFoundES_R = false;


                if(foundF_L){
                    poseEstimationFoundES_L = motionEstimationEssentialMat(inliersF_L1, inliersF_L2, F_L, K_L, KInv_L, T_E_L, R_E_L);
                }

                if(foundF_R){
                    poseEstimationFoundES_R = motionEstimationEssentialMat(inliersF_R1, inliersF_R2, F_R, K_R, KInv_R, T_E_R, R_E_R);
                }
                if (!poseEstimationFoundES_L && !poseEstimationFoundES_R){
                    skipFrame = true;
                    continue;
                    T_E_L = cv::Mat::zeros(3, 1, CV_32F);
                    R_E_L = cv::Mat::eye(3, 3, CV_32F);
                    T_E_R = cv::Mat::zeros(3, 1, CV_32F);
                    R_E_R = cv::Mat::eye(3, 3, CV_32F);
                } else if (!poseEstimationFoundES_L){
                    T_E_L = cv::Mat::zeros(3, 1, CV_32F);
                    R_E_L = cv::Mat::eye(3, 3, CV_32F);
                } else if (!poseEstimationFoundES_R){
                    T_E_R = cv::Mat::zeros(3, 1, CV_32F);
                    R_E_R = cv::Mat::eye(3, 3, CV_32F);
                }

                // find scale factors
                // find right scale factors u und v (according to rodehorst paper)

                // NORMALIZE POINTS
                 // TODO: use inlier!!!
                std::vector<cv::Point2f> normP_L1, normP_R1, normP_L2, normP_R2;
                normalizePoints(KInv_L, KInv_R, points_L1, points_R1, normP_L1, normP_R1);
                normalizePoints(KInv_L, KInv_R, points_L2, points_R2, normP_L2, normP_R2);

                // find scale factors
                // find right scale factors u und v (according to rodehorst paper)
#if 1
                // 1. method:
                float u_L1, u_R1;
                cv::Mat P_L, P_R;
                composeProjectionMat(T_E_L, R_E_L, P_L);
                composeProjectionMat(T_E_R, R_E_R, P_R);
                getScaleFactor(P_0, P_LR, P_L, P_R, normP_L1, normP_R1, normP_L2, normP_R2, u_L1, u_R1);
                if(u_L1 < -1 || u_R1 < -1 || u_L1 > 1000 || u_R1 > 1000 ){
                    std::cout << "scale factors to small or to big:  L: " << u_L1 << "  R: " << u_R1  << std::endl;
                    skipFrame = true;
                    continue;
                }

                T_E_L = T_E_L * u_L1;
                T_E_R = T_E_R * u_R1;

                cout << "u links  1: " << u_L1 << endl;
                cout << "u rechts 1: " << u_R1 << endl << endl;
#else
                // 2. method:
                float u_L2, u_R2;
                getScaleFactor2(T_LR, R_LR, T_E_L, R_E_L, T_E_R, u_L2, u_R2);

                if(u_L2 < -1000 || u_R2 < -1000 || u_L2 > 1000 || u_R2 > 1000 ){
                    std::cout << "scale factors to small or to big:  L: " << u_L2 << "  R: " << u_R2  << std::endl;
                    skipFrame = true;
                    continue;
                }

                T_E_L = T_E_L * u_L2;
                T_E_R = T_E_R * u_R2;

                //compare both methods
                cout << "u links  2: " << u_L2 << endl;
                cout << "u rechts 2: " << u_R2 << endl;
#endif
                //LEFT:
                //rotateRandT(T_E_L, R_E_L);

                cv::Mat newTrans3D_E_L;
                getNewTrans3D( T_E_L, R_E_L, newTrans3D_E_L);


                cv::Mat newPos_ES_L;
                getAbsPos(currentPos_ES_L, newTrans3D_E_L, R_E_L, newPos_ES_L);

                std::stringstream left_ES;
                left_ES << "camera_ES_left" << frame1;

                cv::Mat rotation_ES_L, translation_ES_L;
                decomposeProjectionMat(newPos_ES_L, translation_ES_L, rotation_ES_L);
                //std::cout << "T_ES_left: " << translation_ES_L << std::endl;

                addCameraToVisualizer(translation_ES_L, rotation_ES_L, 255, 0, 0, 20, left_ES.str());


                //RIGHT:
                //rotateRandT(T_E_R, R_E_R);

                cv::Mat newTrans3D_E_R;
                getNewTrans3D( T_E_R, R_E_R, newTrans3D_E_R);

                cv::Mat newPos_ES_R;
                getAbsPos (currentPos_ES_R, newTrans3D_E_R, R_E_R, newPos_ES_R);
                std::stringstream right_ES;
                right_ES << "camera_ES_right" << frame1;

                cv::Mat rotation_ES_R, translation_ES_R;
                decomposeProjectionMat(newPos_ES_R, translation_ES_R, rotation_ES_R);
                //std::cout << "T_ES_right: " << translation_ES_R << std::endl;
                addCameraToVisualizer(translation_ES_R, rotation_ES_R, 0, 255, 0, 20, right_ES.str());

                currentPos_ES_L = newPos_ES_L;
                currentPos_ES_R = newPos_ES_R;


                std::cout << "MOVEMENT FOUND ES: " << T_E_L << "  abs. position  "  << translation_ES_L << std::endl;
                // ##############################################################################
            }

            if (2 == mode) {
                // ################################## PnP #######################################

                // skip frames if there are too less points found
                if (8 > points_L1.size()) {
                    cout << "NO MOVEMENT: to less points found" << endl;
                    skipFrame = true;
                    continue;
                }

                // get inlier from stereo constraints
                std::vector<cv::Point2f> inliersHorizontal_L1, inliersHorizontal_R1, inliersHorizontal_L2, inliersHorizontal_R2;
                getInliersFromHorizontalDirection(make_pair(points_L1, points_R1), inliersHorizontal_L1, inliersHorizontal_R1);
                getInliersFromHorizontalDirection(make_pair(points_L2, points_R2), inliersHorizontal_L2, inliersHorizontal_R2);
                //delete all points that are not correctly found in stereo setup
                deleteZeroLines(points_L1, points_R1, points_L2, points_R2, inliersHorizontal_L1, inliersHorizontal_R1, inliersHorizontal_L2, inliersHorizontal_R2);

                // skip frame because something fails with rectification (ex. frame 287 dbl)
                if (8 > points_L1.size()) {
                    cout << "NO MOVEMENT: couldn't find horizontal points... probably rectification fails or to less feature points found?!" << endl;
                    skipFrame = true;
                    continue;
                }

                // compute fundemental matrix F_L1L2 and get inliers from Ransac
                cv::Mat F_L;
                bool foundF_L;
                std::vector<cv::Point2f> inliersF_L1, inliersF_L2;
                foundF_L = getFundamentalMatrix(points_L1, points_L2, &inliersF_L1, &inliersF_L2, F_L);

                // compute fundemental matrix F_R1R2 and get inliers from Ransac
                cv::Mat F_R;
                bool foundF_R;
                std::vector<cv::Point2f> inliersF_R1, inliersF_R2;
                foundF_R = getFundamentalMatrix(points_R1, points_R2, &inliersF_R1, &inliersF_R2, F_R);

                // make sure that there are all inliers in all frames.
                deleteZeroLines(inliersF_L1, inliersF_L2, inliersF_R1, inliersF_R2);

                drawCorresPoints(image_R1, inliersF_R1, inliersF_R2, "inlier F right " , CV_RGB(0,0,255));
                drawCorresPoints(image_L1, inliersF_L1, inliersF_L2, "inlier F left " , CV_RGB(0,0,255));

                // NORMALIZE POINTS
                std::vector<cv::Point2f> normP_L1, normP_R1, normP_L2, normP_R2;
                normalizePoints(KInv_L, KInv_R, inliersF_L1, inliersF_R1, normP_L1, normP_R1);
                normalizePoints(KInv_L, KInv_R, inliersF_L2, inliersF_R2, normP_L2, normP_R2);

                // TRIANGULATE POINTS
                std::vector<cv::Point3f> pointCloud_1, pointCloud_2;
                TriangulatePointsHZ(P_0, P_LR, normP_L1, normP_R1, 0, pointCloud_1);
                TriangulatePointsHZ(P_0, P_LR, normP_L2, normP_R2, 0, pointCloud_2);


#if 1
                //LEFT:
                bool poseEstimationFoundTemp_L = false;
                cv::Mat T_PnP_L, R_PnP_L;
                if(foundF_L){
                    // GUESS TRANSLATION + ROTATION UP TO SCALE!!!
                    poseEstimationFoundTemp_L = motionEstimationEssentialMat(inliersF_L1, inliersF_L2, F_L, K_L, KInv_L, T_PnP_L, R_PnP_L);
                }

                if (!poseEstimationFoundTemp_L){
                    skipFrame = true;
                    continue;
                }

                // scale factor:
                float u_L1;
                cv::Mat P_L;
                composeProjectionMat(T_PnP_L, R_PnP_L, P_L);
                getScaleFactorLeft(P_0, P_LR, P_L, normP_L1, normP_R1, normP_L2,u_L1);
                if(u_L1 < -1 || u_L1 > 1000 ){
                    std::cout << "scale factors to small or to big:  L: " << u_L1 << std::endl;
                    skipFrame = true;
                    continue;
                }

                T_PnP_L = T_PnP_L * u_L1;

                // use initial guess values for pose estimation
                bool poseEstimationFoundPnP_L = motionEstimationPnP(inliersF_L2, pointCloud_1, K_L, T_PnP_L, R_PnP_L);

                if (!poseEstimationFoundPnP_L){
                    skipFrame = true;
                    continue;
                }

                cv::Mat newTrans3D_PnP_L;
                getNewTrans3D( T_PnP_L, R_PnP_L, newTrans3D_PnP_L);

                cv::Mat newPos_PnP_L;
                getAbsPos(currentPos_PnP_L, newTrans3D_PnP_L, R_PnP_L, newPos_PnP_L);

                cv::Mat rotation_PnP_L, translation_PnP_L;
                decomposeProjectionMat(newPos_PnP_L, translation_PnP_L, rotation_PnP_L);

                std::stringstream left_PnP;
                left_PnP << "camera_PnP_left" << frame1;
                addCameraToVisualizer(translation_PnP_L, rotation_PnP_L, 255, 0, 0, 50, left_PnP.str());
                currentPos_PnP_L  = newPos_PnP_L ;

#else

                //RIGHT:
                bool poseEstimationFoundTemp_R = false;
                cv::Mat  T_PnP_R, R_PnP_R;
                if(foundF_R){
                    // GUESS TRANSLATION + ROTATION UP TO SCALE!!!
                    poseEstimationFoundTemp_R = motionEstimationEssentialMat(inliersF_R1, inliersF_R2, F_R, K_R, KInv_R, T_PnP_R, R_PnP_R);
                }

                if (!poseEstimationFoundTemp_R){
                    skipFrame = true;
                    continue;
                }      

                // use initial guess values for pose estimation
                bool poseEstimationFoundPnP_R = motionEstimationPnP(inliersF_R2, pointCloud_1, K_R, T_PnP_R, R_PnP_R);

                if (!poseEstimationFoundPnP_R){
                    skipFrame = true;
                    continue;
                }

                cv::Mat newTrans3D_PnP_R;
                getNewTrans3D( T_PnP_R, R_PnP_R, newTrans3D_PnP_R);

                cv::Mat newPos_PnP_R;
                getAbsPos(currentPos_PnP_R, newTrans3D_PnP_R, R_PnP_R, newPos_PnP_R);

                cv::Mat rotation_PnP_R, translation_PnP_R;
                decomposeProjectionMat(newPos_PnP_R, translation_PnP_R, rotation_PnP_R);

                std::stringstream right_PnP;
                right_PnP << "camera_PnP_right" << frame1;
                addCameraToVisualizer(translation_PnP_R, rotation_PnP_R, 0, 255, 0, 20, right_PnP.str());
                currentPos_PnP_R  = newPos_PnP_R ;
#endif
                // ##############################################################################
            }

            if (3 == mode) {
                // ################################# STEREO #####################################
                // get inlier from stereo constraints
                std::vector<cv::Point2f> inliersHorizontal_L1, inliersHorizontal_R1, inliersHorizontal_L2, inliersHorizontal_R2;
                getInliersFromHorizontalDirection(make_pair(points_L1, points_R1), inliersHorizontal_L1, inliersHorizontal_R1);
                getInliersFromHorizontalDirection(make_pair(points_L2, points_R2), inliersHorizontal_L2, inliersHorizontal_R2);
                //delete all points that are not correctly found in stereo setup
                deleteZeroLines(points_L1, points_R1, points_L2, points_R2, inliersHorizontal_L1, inliersHorizontal_R1, inliersHorizontal_L2, inliersHorizontal_R2);

                // skip frame because something fails with rectification (ex. frame 287 dbl)
                if (8 > points_L1.size()) {
                    cout << "NO MOVEMENT: couldn't find horizontal points... probably rectification fails or to less feature points found?!" << endl;
                    skipFrame = true;
                    continue;
                }

                // for cv::waitKey input:
                drawCorresPoints(image_L1, points_L1, points_R1, "inlier F1 links rechts", cv::Scalar(255,255,0));
                drawCorresPoints(image_L2, points_L2, points_R2, "inlier F2 links rechts", cv::Scalar(255,255,0));


                // NORMALIZE POINTS
                std::vector<cv::Point2f> normP_L1, normP_R1, normP_L2, normP_R2;
                normalizePoints(KInv_L, KInv_R, points_L1, points_R1, normP_L1, normP_R1);
                normalizePoints(KInv_L, KInv_R, points_L2, points_R2, normP_L2, normP_R2);

                // TRIANGULATE POINTS
                std::vector<cv::Point3f> pointCloud_1, pointCloud_2;
                TriangulatePointsHZ(P_0, P_LR, normP_L1, normP_R1, 0, pointCloud_1);
                TriangulatePointsHZ(P_0, P_LR, normP_L2, normP_R2, 0, pointCloud_2);


                float reproj_error_1L = calculateReprojectionErrorHZ(P_0, normP_L1, pointCloud_1);
                float reproj_error_1R = calculateReprojectionErrorHZ(P_LR, normP_R1, pointCloud_1);

                // check if triangulation success
                if (positionCheck(P_0, pointCloud_1) && positionCheck(P_LR, pointCloud_1) && reproj_error_1L < 1.0 && reproj_error_1R < 1.0 ) {
                    std::cout << "first pointcloud seem's to be not perfect.. take next frame to estimate pos" << std::endl;
                    frame1 = frame2;
                    break;
                }

                float reproj_error_2L = calculateReprojectionErrorHZ(P_0, normP_L2, pointCloud_2);
                float reproj_error_2R = calculateReprojectionErrorHZ(P_LR, normP_R2, pointCloud_2);

                // check if triangulation success
                if (positionCheck(P_0, pointCloud_2) && positionCheck(P_LR, pointCloud_2) && reproj_error_2L < 1.0 && reproj_error_2R < 1.0 ) {
                    std::cout << "second pointcloud seem's to be not perfect.." << std::endl;
                    skipFrame = true;
                    continue;
                }

#if 0
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

                std::vector <cv::Mat_<float>> cloud1;
                std::vector <cv::Mat_<float>> cloud2;
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

#else
                cv::Mat T_Stereo, R_Stereo;
                bool poseEstimationFoundStereo = motionEstimationStereoCloudMatching(pointCloud_1, pointCloud_2, T_Stereo, R_Stereo);
#endif

                if (!poseEstimationFoundStereo){
                    skipFrame = true;
                    continue;
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

                cv::Mat newTrans3D_Stereo;
                getNewTrans3D( T_Stereo, R_Stereo, newTrans3D_Stereo);

                //STEREO:
                cv::Mat newPos_Stereo;
                getAbsPos (currentPos_Stereo, newTrans3D_Stereo, R_Stereo, newPos_Stereo);
                std::stringstream stereo;
                stereo << "camera_Stereo" << frame1;

                cv::Mat rotation, translation;
                decomposeProjectionMat(newPos_Stereo, translation, rotation);
                //std::cout << "T: " << translation << std::endl;

                addCameraToVisualizer(translation, rotation, 0, 0, 255, 100, stereo.str());

                currentPos_Stereo = newPos_Stereo;
                // ##############################################################################
            }

#if 0
            // ######################## TRIANGULATION TEST ################################
            // NORMALIZE POINTS
            std::vector<cv::Point2f> normP_L1, normP_R1, normP_L2, normP_R2;
            normalizePoints(KInv_L, KInv_R, points_L1, points_R1, normP_L1, normP_R1);
            normalizePoints(KInv_L, KInv_R, points_L2, points_R2, normP_L2, normP_R2);

            // TRIANGULATE POINTS
            std::vector<cv::Point3f> pointCloud_1, pointCloud_2;
            TriangulatePointsHZ(P_0, P_LR, normP_L1, normP_R1, 0, pointCloud_1);
            TriangulatePointsHZ(P_0, P_LR, normP_L2, normP_R2, 0, pointCloud_2);


            // STEREO INLIER (POINTS HAVE TO BE LOCATED ON A HORIZONTAL LINE)
            std::vector<cv::Point2f> horizontal_L1, horizontal_R1, horizontal_L2, horizontal_R2;
            getInliersFromHorizontalDirection(make_pair(points_L1, points_R1), horizontal_L1, horizontal_R1);
            getInliersFromHorizontalDirection(make_pair(points_L2, points_R2), horizontal_L2, horizontal_R2);
            deleteZeroLines(horizontal_L1, horizontal_R1, horizontal_L2, horizontal_R2);


            if(0 == horizontal_L1.size()) {
                cout <<  "horizontal inlier: can't find any corresponding points in all 4 frames' "  << std::endl ;
                ++frame;
                continue;
            }

            // NORMALIZE HORIZONTAL POINTS
            std::vector<cv::Point2f> normP_L1_Trian, normP_R1_Trian, normP_L2_Trian, normP_R2_Trian;
            normalizePoints(KInv_L, KInv_R, horizontal_L1, horizontal_R1, normP_L1_Trian, normP_R1_Trian);
            normalizePoints(KInv_L, KInv_R, horizontal_L2, horizontal_R2, normP_L2_Trian, normP_R2_Trian);

            // TRIANGULATE HORIZONTAL POINTS AND GET INLIER
            std::vector<cv::Point3f> pointCloud_inlier_1, pointCloud_inlier_2;
            std::vector<cv::Point2f> inlierTriang_L1, inlierTriang_R1, inlierTriang_L2, inlierTriang_R2;
            TriangulatePointsWithInlier(P_0, P_LR, normP_L1_Trian, normP_R1_Trian, 0, pointCloud_inlier_1, horizontal_L1, horizontal_R1, inlierTriang_L1, inlierTriang_R1);
            TriangulatePointsWithInlier(P_0, P_LR, normP_L2_Trian, normP_R2_Trian, 0, pointCloud_inlier_2, horizontal_L2, horizontal_R2, inlierTriang_L2, inlierTriang_R2);
            deleteZeroLines(inlierTriang_L1, inlierTriang_R1, inlierTriang_L2, inlierTriang_R2, pointCloud_inlier_1, pointCloud_inlier_2);

            if(0 == inlierTriang_L1.size()) {
                cout <<  "triangulation inlier: can't find inlier"  << std::endl ;
                cout <<  "no translation? (triangulation fails by no translation)"  << std::endl ;
                ++frame;
                continue;
            }
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

            rotatePointCloud(pointCloud_inlier_1);
            rotatePointCloud(pointCloud_inlier_2);

            rotatePointCloud(pointCloud_inlier_1, currentPos_ES_L);
            rotatePointCloud(pointCloud_inlier_2, currentPos_ES_L);

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

            AddPointcloudToVisualizer(pointCloud_inlier_1, "cloud1" + std::to_string(frame), RGBValues);
            AddPointcloudToVisualizer(pointCloud_inlier_2, "cloud2" + std::to_string(frame), RGBValues2);

            AddLineToVisualizer(pointCloud_inlier_1, pointCloud_inlier_2, "line"+std::to_string(frame), cv::Scalar(255,0,0));
#endif
        }

        // To Do:
        // swap image files...

        if (1290 < frame1){
            key = cv::waitKey(10);
            if (char(key) == 32) {
                loop = !loop;
            }

            while (loop){
                RunVisualization();

                //to register a event key, you have to make sure that a opencv named Window is open
                key = cv::waitKey(10);
                if (char(key) == 'n') {
                    loop = true;
                    break;
                } else if (char(key) == 32) {
                    loop = false;
                }
            }

        }
    }
    cv::waitKey();
    return 0;
}
