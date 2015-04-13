#ifndef POINTCLOUDVIS_H
#define POINTCLOUDVIS_H

#include "Utility.h"

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <pcl/ros/conversions.h>
#include <vtkPlaneSource.h>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>


#include <vector>
#include <sstream>
#include <utility>
#include <iostream>



void initVisualisation();

void RunVisualization();

void SORFilter();

void PopulatePCLPointCloud(const std::vector<cv::Point3f>& pointcloud,const std::vector<cv::Vec3b>& pointcloud_RGBColor);

inline pcl::PointXYZ Eigen2PointXYZ(Eigen::Vector3f v) { return pcl::PointXYZ(v[0],v[1],v[2]); }
inline pcl::PointXYZRGB Eigen2PointXYZRGB(Eigen::Vector3f v, Eigen::Vector3f rgb) { pcl::PointXYZRGB p(rgb[0],rgb[1],rgb[2]); p.x = v[0]; p.y = v[1]; p.z = v[2]; return p; }
inline pcl::PointNormal Eigen2PointNormal(Eigen::Vector3f v, Eigen::Vector3f n) { pcl::PointNormal p; p.x=v[0];p.y=v[1];p.z=v[2];p.normal_x=n[0];p.normal_y=n[1];p.normal_z=n[2]; return p;}
inline float* Eigen2float6(Eigen::Vector3f v, Eigen::Vector3f rgb) { static float buf[6]; buf[0]=v[0];buf[1]=v[1];buf[2]=v[2];buf[3]=rgb[0];buf[4]=rgb[1];buf[5]=rgb[2]; return buf; }
inline Eigen::Matrix<float,6,1> Eigen2Eigen(Eigen::Vector3f v, Eigen::Vector3f rgb) { return (Eigen::Matrix<float,6,1>() << v[0],v[1],v[2],rgb[0],rgb[1],rgb[2]).finished(); }
inline std::vector<Eigen::Matrix<float,6,1> > AsVector(const Eigen::Matrix<float,6,1>& p1, const Eigen::Matrix<float,6,1>& p2) { 	std::vector<Eigen::Matrix<float,6,1> > v(2); v[0] = p1; v[1] = p2; return v; }

void addCameraToVisualizer(const Eigen::Matrix3f& R, const Eigen::Vector3f& _t, float r, float g, float b, double s = 0.01 /*downscale factor*/, const std::string& name = "");
void addCameraToVisualizer(const float R[9], const float t[3], float r, float g, float b);
void addCameraToVisualizer(const float R[9], const float t[3], float r, float g, float b, double s);
void addCameraToVisualizer(const cv::Vec3f& T, const cv::Matx33f& R, float r, float g, float b, double s, const std::string& name);
void addCameraToVisualizer(const cv::Mat &currentPos, const cv::Mat& T, const cv::Mat &R, float r, float g, float b, double s, const std::string& name);

void AddLineToVisualizer(const std::vector<cv::Point3f>& pointCloud_1, const std::vector<cv::Point3f>& pointCloud_2, std::string name, const cv::Scalar &color);

void AddPointcloudToVisualizer(const std::vector<cv::Point3f>& pointcloud,std::string name,const std::vector<cv::Vec3b>& pointcloud_RGBColor);
#endif // POINTCLOUDVIS_H
