#include "MultiCameraPnP.h"
#include "FindCameraMatrices.h"

// find pose estimation using orientation of pointcloud
bool findPoseEstimation(
        cv::Mat const& P,
        std::vector<cv::Point3f> const& ppcloud,
        std::vector<cv::Point2f> const& normPoints, //(normalized)
        cv::Mat& T,
        cv::Mat& R
        )
{
    if(ppcloud.size() <= 7 || normPoints.size() <= 7 || ppcloud.size() != normPoints.size()) {
        //something went wrong aligning 3D to 2D points..
        cerr << "couldn't find [enough] corresponding cloud points... (only " << ppcloud.size() << ")" <<endl;
        return false;
    }
    vector<int> inliers;

    double minVal,maxVal;
    cv::minMaxIdx(normPoints,&minVal,&maxVal);

    cv::Mat K = cv::Mat::eye(3, 3, CV_64F); // use identity calibration Mat (points are allready normalised)
    vector<double > distCoeffVec; //just use empty vector.. images are allready undistorted..
    cv::Mat_<double> rvec;
    cv::solvePnPRansac(ppcloud, normPoints, K, distCoeffVec, rvec, T, true, 1000, 0.006 * maxVal, 0.25 * (double)(normPoints.size()), inliers, CV_EPNP);


    // calculate reprojection error and define inliers
    for (unsigned int i = 0; i < ppcloud.size(); ++i) {
        // reproject 3d points
        cv::Mat_<double> point3D_h(4, 1);
        point3D_h(0) = ppcloud[i].x;
        point3D_h(1) = ppcloud[i].y;
        point3D_h(2) = ppcloud[i].z;
        point3D_h(3) = 1.0;

        // reproject points
        cv::Mat_<double> reprojectedPoint_h = P * point3D_h;

        // convert reprojected image point to carthesian coordinates
        cv::Point2f reprojectedPoint(reprojectedPoint_h(0) / reprojectedPoint_h(2), reprojectedPoint_h(1) / reprojectedPoint_h(2));

        double reproj_error = (cv::norm(normPoints[i] - reprojectedPoint));
        if (reproj_error < 10.0) {
            inliers.push_back(i);
        }
    }

    if(inliers.size() < (double)(normPoints.size())/5.0) {
        cerr << "not enough inliers to consider a good pose ("<<inliers.size()<<"/"<<normPoints.size()<<")"<< endl;
        return false;
    }

    if(cv::norm(T) > 2000.0) {
        // this is bad...
        cerr << "estimated camera movement is too big, skip this camera\r\n";
        return false;
    }

    cv::Rodrigues(rvec, R);

    if(!CheckCoherentRotation(R)) {
        cerr << "rotation is incoherent. we should try a different base view..." << endl;
        return false;
    }
    std::cout << "found t = " << T << "\nR = \n"<< R <<std::endl;
    return true;
}
