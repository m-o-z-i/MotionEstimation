#ifndef POINTCLOUDVIS_H
#define POINTCLOUDVIS_H

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <pcl/ros/conversions.h>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>


void RunVisualization(const std::vector<cv::Point3f>& pointcloud, const std::vector<cv::Vec3b>& pointcloud_RGB = std::vector<cv::Vec3b>());

void SORFilter();

void PopulatePCLPointCloud(const std::vector<cv::Point3f>& pointcloud,const std::vector<cv::Vec3b>& pointcloud_RGBColor);


#endif // POINTCLOUDVIS_H
