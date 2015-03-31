#include "PointCloudVis.h"


pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr orig_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

void PopulatePCLPointCloud(const std::vector<cv::Point3f> &pointcloud,
                           const std::vector<cv::Vec3b>& pointcloud_RGBColor
                           )
{
    cout<<"Creating point cloud...";
    cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);

    for (unsigned int i=0; i<pointcloud.size(); ++i) {
        // get the RGB color value for the point
        cv::Vec3b rgbv(255,255,255);
        if (pointcloud_RGBColor.size() >= i) {
            rgbv = pointcloud_RGBColor[i];
        }

        // check for erroneous coordinates (NaN, Inf, etc.)
        if (pointcloud[i].x != pointcloud[i].x || isnan(pointcloud[i].x) ||
                pointcloud[i].y != pointcloud[i].y || isnan(pointcloud[i].y) ||
                pointcloud[i].z != pointcloud[i].z || isnan(pointcloud[i].z) ){
            continue;
        }

        pcl::PointXYZRGB pclp;

        // 3D coordinates
        pclp.x = pointcloud[i].x;
        pclp.y = pointcloud[i].y;
        pclp.z = pointcloud[i].z;


        // RGB color, needs to be represented as an integer
        uint32_t rgb = ((uint32_t)rgbv[2] << 16 | (uint32_t)rgbv[1] << 8 | (uint32_t)rgbv[0]);
        pclp.rgb = *reinterpret_cast<float*>(&rgb);

        cloud->push_back(pclp);
    }

    cloud->width = (uint32_t) cloud->points.size(); // number of points
    cloud->height = 1; // a list of points, one row of data
}

void SORFilter() {

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);

    std::cerr<<"Cloud before SOR filtering: "<< cloud->width * cloud->height <<" data points"<<std::endl;


    // Create the filtering object
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB>sor;
    sor.setInputCloud (cloud);
    sor.setMeanK (50);
    sor.setStddevMulThresh (1.0);
    sor.filter (*cloud_filtered);

    std::cerr<<"Cloud after SOR filtering: "<<cloud_filtered->width * cloud_filtered->height <<" data points "<<std::endl;

    pcl::copyPointCloud(*cloud_filtered,*cloud);
}

void RunVisualization(const std::vector<cv::Point3f>& pointcloud,
                      const std::vector<cv::Vec3b>& pointcloud_RGBColor) {
    PopulatePCLPointCloud(pointcloud,pointcloud_RGBColor);
    //SORFilter();
    pcl::copyPointCloud(*cloud,*orig_cloud);

    pcl::visualization::CloudViewer viewer("Cloud Viewer");

    // run the cloud viewer
    viewer.showCloud(orig_cloud,"orig");

    while (!viewer.wasStopped ())
    {
        // NOP
    }
}


