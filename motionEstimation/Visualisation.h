#ifndef VISUALISATION_H
#define VISUALISATION_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

using namespace std;

static const float pi = 3.14159265358979323846;


void drawLine(IplImage* ref, cv::Point2f p, cv::Point2f q, float angle, cv::Scalar const& color = CV_RGB(0,0,0), int line_thickness = 1);
void drawLine(cv::Mat& ref, cv::Point2f p, cv::Point2f q, float angle, cv::Scalar const& color = CV_RGB(0,0,0), int line_thickness = 1);
void drawPoints (cv::Mat image, vector<cv::Point2f> points, string windowName, cv::Scalar const& color = CV_RGB(0,0,0));

void drawEpipolarLines(cv::Mat frame1, vector<cv::Point2f> const& points1, cv::Mat F);
void drawHomographyPoints(cv::Mat frame1, cv::Mat frame2, vector<cv::Point2f> const& points1, vector<cv::Point2f> const& points2);

void drawCorresPoints(const cv::Mat &image, const vector<cv::Point2f> &inliers1, const vector<cv::Point2f> &inliers2, string name, cv::Scalar const& color);
void drawCorresPointsRef(cv::Mat& image, const vector<cv::Point2f>& inliers1, const vector<cv::Point2f>& inliers2, string name, cv::Scalar const& color);

void drawOptFlowMap (cv::Mat flow, cv::Mat& cflowmap, int step, const cv::Scalar& color);

void drawAllStuff (cv::Mat mat_image11, cv::Mat mat_image12, cv::Mat mat_image21, cv::Mat mat_image22, int frame);

cv::Point2f drawCameraPath(cv::Mat& img, const cv::Point2f prevPos, const cv::Mat& T, string name, cv::Scalar const& color);

#endif // VISUALISATION_H
