#ifndef TRIANGULATION_H
#define TRIANGULATION_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;

#define EPSILON 0.0001


cv::Mat_<float> LinearLSTriangulation(cv::Point3f u,cv::Matx34f P,
                                       cv::Point3f u1, cv::Matx34f P1);

void TriangulatePointsHZ(const cv::Mat& P_L, const cv::Mat& P_R,
                         const vector<cv::Point2f>& points1, const vector<cv::Point2f>& points2,
                         int numberOfTriangulations,
                         vector<cv::Point3f>& pointcloud);

void TriangulateOpenCV(const cv::Mat& P_L,
                       const cv::Mat& P_R,
                       const vector<cv::Point2f>& points_L,
                       const vector<cv::Point2f>& points_R,
                       std::vector<cv::Point3f>& outCloud);

cv::Mat_<float> IterativeLinearLSTriangulation(cv::Point3f point2d1_h, cv::Matx34f P0,
                                                cv::Point3f point2d2_h, cv::Matx34f P1);

void triangulate(const cv::Mat& P0, const cv::Mat& P1,
                 const vector<cv::Point2f>& x0, const vector<cv::Point2f>& x1,
                 vector<cv::Point3f>& result3D);

void TriangulatePointsWithInlier(const cv::Matx34f& P0, const cv::Matx34f& P1,
        const vector<cv::Point2f>& points1,
        const vector<cv::Point2f>& points2,
        int numberOfTriangulations,
        vector<cv::Point3f>& pointcloud,
        vector<cv::Point2f>& inlier1,
        vector<cv::Point2f>& inlier2);


void computeReprojectionError(const cv::Mat& P,
                              const vector<cv::Point2f>& points,
                              const vector<cv::Point3f>& worldCoordinates,
                              vector<cv::Point3f>& pReprojected,
                              vector<cv::Point2f>& reprojectionErrors,
                              cv::Point2f& avgReprojectionError);

float calculateReprojectionErrorOpenCV(const cv::Mat& P,
                                        const cv::Mat& K, const cv::Mat distCoeff,
                                        const vector<cv::Point2f>& points2D,
                                        const std::vector<cv::Point3f>& points3D);

float calculateReprojectionErrorHZ(const cv::Mat& P,
                                    const vector<cv::Point2f> &points2D,
                                    const std::vector<cv::Point3f>& points3D);

#endif // TRIANGULATION_H
