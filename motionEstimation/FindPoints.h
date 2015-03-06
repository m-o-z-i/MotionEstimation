#ifndef FINDPOINTS_H
#define FINDPOINTS_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;

std::vector<cv::Point2f> getStrongFeaturePoints (cv::Mat const& image, int number = 50, float minQualityLevel = .03, float minDistance = 0.1);
pair<vector<cv::Point2f>, vector<cv::Point2f> > refindFeaturePoints(cv::Mat const& prev_image, cv::Mat const& next_image, vector<cv::Point2f> frame1_features);

void getInliersFromMeanValue (pair<vector<cv::Point2f>, vector<cv::Point2f>> const& features, vector<cv::Point2f> *inliers2, vector<cv::Point2f> *inliers1);

#endif // FINDPOINTS_H