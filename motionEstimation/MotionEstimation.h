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

bool motionEstimationEssentialMat (const cv::Mat& image1, const cv::Mat& image2,
                                   const std::vector<cv::Point2f>& points1,
                                   const std::vector<cv::Point2f>& points2,
                                   const cv::Mat& K, const cv::Mat& KInv,
                                   cv::Mat& T, cv::Mat& R);

bool motionEstimationPnP (const std::vector<cv::Point2f>& points_2,
                          const std::vector<cv::Point3f>& pointCloud_LR,
                          const cv::Mat& K,
                          cv::Mat& T, cv::Mat& R);

bool TestTriangulation(const std::vector<cv::Point2f>& points_L1,
                        const std::vector<cv::Point2f>& points_R1,
                        const cv::Mat& P_LR, const cv::Mat& KInv_L,
                        const cv::Mat& KInv_R);


#endif // MOTIONESTIMATION_H
