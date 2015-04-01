#ifndef MULTICAMERAPNP_H
#define MULTICAMERAPNP_H

#include <vector>
#include <opencv2/core/core.hpp>
#include "Triangulation.h"

using namespace std;

bool findPoseEstimation(const cv::Mat &P, cv::Mat const& K, const std::vector<cv::Point3f> &ppcloud, const std::vector<cv::Point2f> &normPoints, cv::Mat &T, cv::Mat &R);

#endif // MULTICAMERAPNP_H
