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

void getInliersFromMedianValue (pair<vector<cv::Point2f>, vector<cv::Point2f>> const& features, vector<cv::Point2f> *inliers2, vector<cv::Point2f> *inliers1);
void deleteUnvisiblePoints(pair<vector<cv::Point2f>, vector<cv::Point2f>>& corresPoints1to2, pair<vector<cv::Point2f>, vector<cv::Point2f> >& corresPointsL1toR1, pair<vector<cv::Point2f>, vector<cv::Point2f> >& corresPointsL2toR2, int resX, int resY);
void deleteZeroLines(vector<cv::Point2f> &points1, vector<cv::Point2f> &points2);
void deleteZeroLines(vector<cv::Point2f>& points1La, vector<cv::Point2f>& points1Lb, vector<cv::Point2f>& points1Ra, vector<cv::Point2f>& points1Rb, vector<cv::Point2f>& points2L, vector<cv::Point2f>& points2R);

void normalizePoints(const cv::Mat& KLInv, const vector<cv::Point2f>& inliersFL1, const cv::Mat& KRInv, const vector<cv::Point2f>& inliersFR1, vector<cv::Point2f>& normPointsL, vector<cv::Point2f>& normPointsR);





#endif // FINDPOINTS_H
