#ifndef MULTICAMERAPNP_H
#define MULTICAMERAPNP_H

#include <vector>
#include <opencv2/core/core.hpp>

using namespace std;

bool findPoseEstimation(cv::Mat_<double>& rvec, cv::Mat &t, cv::Mat_<double>& R, std::vector<cv::Point3f> ppcloud, std::vector<cv::Point2f> imgPoints, cv::Mat K);

#endif // MULTICAMERAPNP_H
