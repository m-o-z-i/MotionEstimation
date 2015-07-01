#ifndef MOTIONESTIMATION_H
#define MOTIONESTIMATION_H

#include "FindCameraMatrices.h"
#include "FindPoints.h"
#include "Triangulation.h"
#include "Visualisation.h"
#include "PointCloudVis.h"
#include "Utility.h"

#include <cmath>
#include <math.h>
#include <vector>
#include <utility>
#include <stack>
#include <sstream>
#include <string.h>
#include <dirent.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>


bool motionEstimationStereoCloudMatching (const std::vector<cv::Point3f>& pointCloud_1,
                                          const std::vector<cv::Point3f>& pointCloud_2,
                                          cv::Mat& T, cv::Mat& R);

bool motionEstimationEssentialMat (const std::vector<cv::Point2f>& points_1,
                                   const std::vector<cv::Point2f>& points_2,
                                   const cv::Mat& F,
                                   const cv::Mat& K,
                                   cv::Mat& T, cv::Mat& R);

bool motionEstimationPnP (const std::vector<cv::Point2f>& imgPoints,
                          const std::vector<cv::Point3f>& pointCloud_1LR,
                          const cv::Mat& K,
                          cv::Mat& T, cv::Mat& R);

#endif // MOTIONESTIMATION_H
