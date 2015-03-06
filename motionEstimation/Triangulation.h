#ifndef TRIANGULATION_H
#define TRIANGULATION_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;

#define EPSILON 0.0001


cv::Mat_<double> LinearLSTriangulation(cv::Point3d u,cv::Matx34d P, cv::Point3d u1, cv::Matx34d P1);
double TriangulatePoints(const vector<cv::Point2f>& points1, const vector<cv::Point2f>& points2, const cv::Mat& K, const cv::Mat&Kinv, const cv::Matx34f& P0, const cv::Matx34f& P1, vector<cv::Point3f>& pointcloud);
double TriangulateOpenCV(const cv::Mat K, const cv::Mat distCoeff, const vector<cv::Point2f>& inliersF1, const vector<cv::Point2f>& inliersF2, cv::Mat& P0, cv::Mat& P1, std::vector<cv::Point3f>& outCloud);
cv::Mat_<double> IterativeLinearLSTriangulation(cv::Point3d point2d1_h, cv::Matx34d P0, cv::Point3d point2d2_h, cv::Matx34d P1);
void triangulate(cv::Mat& P0, cv::Mat& P1, vector<cv::Point2f>& x0, vector<cv::Point2f>& x1, vector<cv::Point3f>& result3D);
void computeReprojectionError(cv::Mat& P, vector<cv::Point2f>& p, vector<cv::Point3f>& worldCoordinates, vector<cv::Point3f>& pReprojected, vector<cv::Point2f>& reprojectionErrors, cv::Point2f& avgReprojectionError);


#endif // TRIANGULATION_H