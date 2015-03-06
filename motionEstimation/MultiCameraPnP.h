#ifndef MULTICAMERAPNP_H
#define MULTICAMERAPNP_H

#include <vector>
#include <opencv2/core/core.hpp>

using namespace std;

bool findPoseEstimation(cv::Mat_<double>& rvec, cv::Mat_<double>& t, cv::Mat_<double>& R, std::vector<cv::Point3f> ppcloud, std::vector<cv::Point2f> imgPoints, cv::Mat K, cv::Mat distortion_coeff);

#endif // MULTICAMERAPNP_H
