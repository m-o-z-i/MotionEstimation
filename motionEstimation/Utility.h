#ifndef UTILITY_H
#define UTILITY_H

#include <dirent.h>
#include <vector>
#include <opencv2/opencv.hpp>



int getFiles (std::string const& dir, std::vector<std::string> &files);

void getAbsPos (cv::Mat const& currentPos, cv::Mat const& T, cv::Mat const& R, cv::Mat& newPos);
void getNewTrans3D (cv::Mat const& T, cv::Mat const& R, cv::Mat &newTrans);

void decomposeProjectionMat(const cv::Mat& P, cv::Mat& T, cv::Mat& R);
void composeProjectionMat(const cv::Mat &T, const cv::Mat& R, cv::Mat& P);

void rotatePointCloud(std::vector<cv::Point3f> &cloud);
void rotatePointCloud(std::vector<cv::Point3f>& cloud, const cv::Mat P);

void KeyPointsToPoints(const std::vector<cv::KeyPoint>& kps, std::vector<cv::Point2f>& ps);

void PointsToKeyPoints(const std::vector<cv::Point2f>& ps, std::vector<cv::KeyPoint>& kps);

void decomposeRotMat(const cv::Mat& R, float& x, float& y, float& z);
bool calcCoordinate(cv::Mat_<float> &toReturn,cv::Mat const& Q, cv::Mat const& disparityMap,int x,int y);


void rotateRandT(cv::Mat& Trans, cv::Mat& Rot);

#endif // UTILITY_H
