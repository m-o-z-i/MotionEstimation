#ifndef FINDCAMERAMATRICES_H
#define FINDCAMERAMATRICES_H


#include <vector>
#include <utility>

#include <opencv2/opencv.hpp>


using namespace std;

bool TestTriangulation(const cv::Matx34f& P, const vector<cv::Point3f>& points3D);
void getFundamentalMatrix(pair<vector<cv::Point2f>, vector<cv::Point2f>> const& points, vector<cv::Point2f> *inliers1, vector<cv::Point2f> *inliers2, cv::Mat& F);
bool CheckCoherentRotation(const cv::Mat& R);
bool DecomposeEtoRandT(const cv::Mat& E, cv::Mat_<double>& R1, cv::Mat_<double>& R2, cv::Mat_<double>& t1, cv::Mat_<double>& t2);
bool getRightProjectionMat(cv::Mat& E, const cv::Mat K, const cv::Mat KInv, const cv::Mat distCoeff, cv::Mat& P1, const vector<cv::Point2f>& points2D_1, const vector<cv::Point2f>& points2D_2, std::vector<cv::Point3f>& outCloud);
void loadIntrinsic(std::string name, cv::Mat& K, cv::Mat& distCoeff);
void loadExtrinsic(cv::Mat& R, cv::Mat& T, cv::Mat& E, cv::Mat& F );

#endif // FINDCAMERAMATRICES_H
