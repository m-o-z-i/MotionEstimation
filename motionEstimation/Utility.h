#ifndef UTILITY_H
#define UTILITY_H

#include <dirent.h>
#include <vector>
#include <opencv2/opencv.hpp>



int getFiles (std::string const& dir, std::vector<std::string> &files);
void getNewPos (cv::Mat const& currentPos, cv::Mat const& T, cv::Mat const& R, cv::Mat& newPos);
void decomposeProjectionMat(const cv::Mat& P, cv::Mat& T, cv::Mat& R);
void composeProjectionMat(const cv::Mat &T, const cv::Mat& R, cv::Mat& P);

#endif // UTILITY_H
