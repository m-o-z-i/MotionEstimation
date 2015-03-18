#ifndef FINDCAMERAMATRICES_H
#define FINDCAMERAMATRICES_H


#include <vector>
#include <utility>

#include <opencv2/opencv.hpp>


using namespace std;

bool TestTriangulation(const cv::Matx34f& P, const vector<cv::Point3f>& points3D);
bool getFundamentalMatrix(pair<vector<cv::Point2f>, vector<cv::Point2f>> const& points, vector<cv::Point2f> *inliers1, vector<cv::Point2f> *inliers2, cv::Mat& F);
bool CheckCoherentRotation(const cv::Mat& R);
bool DecomposeEtoRandT(const cv::Mat& E, cv::Mat_<double>& R1, cv::Mat_<double>& R2, cv::Mat_<double>& t1, cv::Mat_<double>& t2);
bool getRightProjectionMat(cv::Mat& E, cv::Mat& P1, const vector<cv::Point2f> &normPoints2D_L, const vector<cv::Point2f> &normPoints2D_R, std::vector<cv::Point3f>& outCloud);
void loadIntrinsic(std::string name, cv::Mat& K, cv::Mat& distCoeff);
void loadExtrinsic(cv::Mat& R, cv::Mat& T, cv::Mat& E, cv::Mat& F );
void decomposeProjectionMat(const cv::Mat& P, cv::Mat& R, cv::Mat& T);

void getScaleFactor(const cv::Mat& P0, const cv::Mat& P_LR, const cv::Mat& P_L, const cv::Mat& P_R, const vector<cv::Point2f>& normPoints_L1, const vector<cv::Point2f>&normPoints_R1, const vector<cv::Point2f>&normPoints_L2, const vector<cv::Point2f>& normPoints_R2, double& u, double& v);
void getScaleFactor2(const cv::Mat& T_L, const cv::Mat &R_L, const cv::Mat& T_R, const cv::Mat &T_LR, const cv::Mat &R_LR, double& u, double& v);
#endif // FINDCAMERAMATRICES_H
