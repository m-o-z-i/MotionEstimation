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

    vector<int> inliers;

    double minVal,maxVal;
    cv::minMaxIdx(imgPoints,&minVal,&maxVal);

    vector<float > distCoeffVec; //just use empty vector.. images are allready undistorted..

    //can't work.. a cloud and points...?!
    cv::solvePnPRansac(pointCloud_1LR, imgPoints, K, distCoeffVec, rvec, T, true, 1000, 0.006 * maxVal, 0.25 * (float)(imgPoints.size()), inliers, CV_EPNP);

    // calculate reprojection error and define inliers
    std::vector<cv::Point2f> projected3D;
    cv::projectPoints(pointCloud_1LR, rvec, T, K, distCoeffVec, projected3D);
    if(inliers.size()==0) { //get inliers
        for(unsigned int i=0;i<projected3D.size();i++) {
            if(norm(projected3D[i]-imgPoints[i]) < 10.0)
                inliers.push_back(i);
        }
    }

    if(inliers.size() < (float)(imgPoints.size())/5.0) {
        cerr << "NO MOVEMENT: not enough inliers to consider a good pose ("<<inliers.size()<<"/"<<imgPoints.size()<<")"<< endl;
        return false;
    }

    if(cv::norm(T) > 2000.0) {
        // this is bad...
        cerr << "NO MOVEMENT: estimated camera movement is too big, skip this camera\r\n";
        return false;
    }

    cv::Rodrigues(rvec, R);
    if(!CheckCoherentRotation(R)) {
        cerr << "NO MOVEMENT: rotation is incoherent. we should try a different base view..." << endl;
        return false;
    }

    return true;
}


bool motionEstimationEssentialMat (const std::vector<cv::Point2f>& inliersF1,
                                   const std::vector<cv::Point2f>& inliersF2,
                                   const cv::Mat& F,
                                   const cv::Mat& K, const cv::Mat& KInv,
                                   cv::Mat& T, cv::Mat& R)
{
    // normalisize all Points
    std::vector<cv::Point2f> normPoints1, normPoints2;
    normalizePoints(KInv, inliersF1, inliersF2, normPoints1, normPoints2);

    // calculate essential mat
    cv::Mat E = K.t() * F * K; //according to HZ (9.12)

    // decompose right solution for R and T values and saved it to P1. get point cloud of triangulated points
    cv::Mat P;
    std::vector<cv::Point3f> pointCloud;
    bool goodPFound = getRightProjectionMat(E, P, normPoints1, normPoints2, pointCloud);

    if (!goodPFound) {
        cout << "NO MOVEMENT: no perspective Mat Found" << endl;
        return false;
    }

    cv::Mat T_temp, R_temp;
    decomposeProjectionMat(P, T_temp, R_temp);

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









