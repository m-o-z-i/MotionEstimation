#include "MotionEstimation.h"


// find pose estimation using orientation mapping of pointcloud with ransac
bool motionEstimationPnP (const std::vector<cv::Point2f>& imgPoints,
                          const std::vector<cv::Point3f>& pointCloud_1LR,
                          const cv::Mat& K,
                          cv::Mat& T, cv::Mat& R)
{
    if(pointCloud_1LR.size() <= 7 || imgPoints.size() <= 7 || pointCloud_1LR.size() != imgPoints.size()) {
        //something went wrong aligning 3D to 2D points..
        cerr << "NO MOVEMENT: couldn't find [enough] corresponding cloud points... (only " << pointCloud_1LR.size() << ")" <<endl;
        return false;
    }

    cv::Mat rvec;
    cv::Rodrigues(R, rvec);

    std::vector<int> inliers;

    double minVal,maxVal;
    cv::minMaxIdx(imgPoints,&minVal,&maxVal);

    std::vector<float > distCoeffVec; //just use empty vector.. images are allready undistorted..

    /*
     * solvePnPRansac(InputArray objectPoints, InputArray imagePoints, InputArray cameraMatrix, InputArray distCoeffs,
     *                OutputArray rvec, OutputArray tvec, bool useExtrinsicGuess=false, int iterationsCount=100,
     *                float reprojectionError=8.0, int minInliersCount=100, OutputArray inliers=noArray(), int flags=ITERATIVE )
     */
    //cv::solvePnPRansac(pointCloud_1LR, imgPoints, K, distCoeffVec, rvec, T);
    cv::solvePnPRansac(pointCloud_1LR, imgPoints, K, distCoeffVec, rvec, T, true, 1000, 0.006 * maxVal, 0.25 * (float)(imgPoints.size()), inliers, CV_EPNP);
    rvec.convertTo(rvec, CV_32F);
    T.convertTo(T, CV_32F);

    // calculate reprojection error and define inliers
    std::vector<cv::Point2f> projected3D;
    cv::projectPoints(pointCloud_1LR, rvec, T, K, distCoeffVec, projected3D);
    if(inliers.size() == 0){

        for(unsigned int i=0;i<projected3D.size();i++) {
            //std::cout << "repro error " << norm(projected3D[i]-imgPoints[i])  << std::endl;
            if(norm(projected3D[i]-imgPoints[i]) < 10.0)
                inliers.push_back(i);
        }
    }


    if(inliers.size() < (float)(imgPoints.size())/5.0) {
        std::cout << "NO MOVEMENT: not enough inliers to consider a good pose ("<<inliers.size()<<"/"<<imgPoints.size()<<")" << std::endl;
        return false;
    }

    cv::Rodrigues(rvec, R);
    R.convertTo(R, CV_32F);
    if(!CheckCoherentRotation(R)) {
        std::cout <<  "NO MOVEMENT: rotation is incoherent..." << std::endl;
        return false;
    }

    return true;
}


bool motionEstimationEssentialMat (const std::vector<cv::Point2f>& points_1,
                                   const std::vector<cv::Point2f>& points_2,
                                   const cv::Mat& F,
                                   const cv::Mat& K,
                                   cv::Mat& T, cv::Mat& R)
{
    // calculate essential mat
    cv::Mat E = K.t() * F * K; //according to HZ (9.12)

    // decompose right solution for R and T values and saved it to P1. get point cloud of triangulated points
    cv::Mat P;
    std::vector<cv::Point3f> pointCloud;
    bool goodPFound = getRightProjectionMat(E, P, K, points_1, points_2, pointCloud);

    if (!goodPFound) {
        cout << "NO MOVEMENT: no perspective Mat Found" << endl;
        return false;
    }

    cv::Mat T_temp, R_temp;
    decomposeProjectionMat(P, T_temp, R_temp);

    //delete y motion
    //T_temp.at<float>(1)=0;
    T = T_temp;
    R = R_temp;

    return true;
}



bool motionEstimationStereoCloudMatching (const std::vector<cv::Point3f>& pointCloud_1,
                                          const std::vector<cv::Point3f>& pointCloud_2,
                                          cv::Mat& T, cv::Mat& R)
{
    int numberPts = pointCloud_1.size();

    if (3 >= numberPts) {
        cout << "to less 3d points" << endl;
        return false;
    }

    // convert to normalized homogenous coordinates.... WHYY?????
    cv::Mat X_1h, X_2h;
    cv::convertPointsToHomogeneous(cv::Mat(pointCloud_1), X_1h);
    cv::convertPointsToHomogeneous(cv::Mat(pointCloud_2), X_2h);

    //estimate rotation
    //1. translate 3d points by theire mean vectors to origin
    cv::Point3f Xmean_1, Xmean_2;
    for (unsigned int i = 0; i < numberPts; ++i){
        Xmean_1 += pointCloud_1[i];
        Xmean_2 += pointCloud_2[i];
    }

    Xmean_1 = 1.0/numberPts * Xmean_1;
    Xmean_2 = 1.0/numberPts * Xmean_2;

    std::vector<cv::Point3f> XC_1, XC_2;
    for (unsigned int i = 0; i < numberPts; ++i){
        XC_1.push_back( pointCloud_1[i] - Xmean_1);
        XC_2.push_back( pointCloud_2[i] - Xmean_2);
    }


    //2. compute 3x3 covariance matrix
    cv::Mat A_temp;
    for (unsigned int i = 0; i < numberPts; ++i){
        A_temp = A_temp + (cv::Mat(XC_2[i])*cv::Mat(XC_1[i]).t());
    }

    cv::Mat A = 1.0/numberPts * A_temp;

    // estimate R from svd(A)
    cv::SVD svd(A, cv::SVD::FULL_UV);
    cv::Mat svd_u = svd.u;
    cv::Mat svd_s = svd.w;
    cv::Mat svd_v = svd.vt;

    cv::Mat S_diag = cv::Mat::eye(3, 3, CV_32F); // 3 x 3 mat

    if (cv::determinant(svd_u) * cv::determinant(svd_v) < 0){ // det(v) == det(v.t())?
        S_diag.at<float>(8) = -1;
    }

    cv::Mat R_temp = svd_u * S_diag * svd_v;

    // compute translation
    cv::Mat T_temp = cv::Mat(Xmean_2) - R_temp*cv::Mat(Xmean_1);

    T = T_temp;
    R = R_temp;

    return true;
}









