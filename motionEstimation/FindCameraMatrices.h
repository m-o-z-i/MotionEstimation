#ifndef FINDCAMERAMATRICES_H
#define FINDCAMERAMATRICES_H

#include "Utility.h"

#include <vector>
#include <utility>

#include <opencv2/opencv.hpp>


using namespace std;

bool positionCheck(const cv::Matx34f& P, const vector<cv::Point3f>& points3D);
bool getFundamentalMatrix(const vector<cv::Point2f> &points1, const vector<cv::Point2f> &points2, vector<cv::Point2f> *inliers1, vector<cv::Point2f> *inliers2, cv::Mat& F);
bool CheckCoherentRotation(const cv::Mat& R);
bool DecomposeEtoRandT(const cv::Mat& E, cv::Mat_<float>& R1, cv::Mat_<float>& R2, cv::Mat_<float>& t1, cv::Mat_<float>& t2);

bool getRightProjectionMat(cv::Mat& E,
                            cv::Mat& P1, const cv::Mat &K,
                            const vector<cv::Point2f>& points2D_1,
                            const vector<cv::Point2f>& points2D_2,
                            std::vector<cv::Point3f>& outCloud);

void loadIntrinsic(string path, cv::Mat& K_L, cv::Mat& K_R, cv::Mat &distCoeff_L, cv::Mat &distCoeff_R);
void loadExtrinsic(string path, cv::Mat& R, cv::Mat& T, cv::Mat& E, cv::Mat& F );

void getScaleFactor(const cv::Mat& P0, const cv::Mat& P_LR, const cv::Mat& P_L, const cv::Mat& P_R, const vector<cv::Point2f>& points_L1, const vector<cv::Point2f>&points_R1, const vector<cv::Point2f>&points_L2, const vector<cv::Point2f>& points_R2, float& u, float& v, std::vector<cv::Point3f> &pCloud, std::vector<cv::Point3f> &nearestPoints);
void getScaleFactorRight(const cv::Mat& P0, const cv::Mat& P_LR, const cv::Mat& P_R,
                    const std::vector<cv::Point2f>& points_L1, const std::vector<cv::Point2f>& points_R1,
                    const std::vector<cv::Point2f>& points_R2,
                    float& u);
void getScaleFactorLeft(const cv::Mat& P0, const cv::Mat& P_LR, const cv::Mat& P_L,
                    const std::vector<cv::Point2f>& points_L1, const std::vector<cv::Point2f>& points_R1,
                    const std::vector<cv::Point2f>& points_L2,
                    float& u);

void getScaleFactor2(const cv::Mat& T_LR, const cv::Mat& R_LR, const cv::Mat& T_L, const cv::Mat& R_L, const cv::Mat& T_R,  float& u, float& v);


#endif // FINDCAMERAMATRICES_H
