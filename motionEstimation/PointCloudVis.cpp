#include "PointCloudVis.h"


pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr orig_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

////////////////////////////////// Show Camera ////////////////////////////////////
std::deque<std::pair<std::string,pcl::PolygonMesh> >					cam_meshes;
std::deque<std::pair<std::string,std::vector<Eigen::Matrix<float,6,1> > > >	linesToShow;
//TODO define mutex
bool							bShowCam;
int								iCamCounter = 0;
int								iLineCounter = 0;
int								ipolygon[18] = {0,1,2,  0,3,1,  0,4,3,  0,2,4,  3,1,4,   2,4,1};



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

    //pcl::visualization::CloudViewer viewer("Cloud Viewer");
    pcl::visualization::PCLVisualizer viewer("Cloud Viewer");

    // run the cloud viewer
    viewer.addPointCloud(orig_cloud,"orig");

    if(cam_meshes.size() > 0) {
        int num_cams = cam_meshes.size();
        cout << "showing " << num_cams << " cameras" << endl;
        while(cam_meshes.size()>0) {
            viewer.removeShape(cam_meshes.front().first);
            viewer.addPolygonMesh(cam_meshes.front().second,cam_meshes.front().first);
            cam_meshes.pop_front();
        }
    }
    if(linesToShow.size() > 0) {
        cout << "showing " << linesToShow.size() << " lines" << endl;
        while(linesToShow.size()>0) {
            std::vector<Eigen::Matrix<float,6,1> > oneline = linesToShow.front().second;
            pcl::PointXYZRGB	A(oneline[0][3],oneline[0][4],oneline[0][5]),
                    B(oneline[1][3],oneline[1][4],oneline[1][5]);
            for(int j=0;j<3;j++) {A.data[j] = oneline[0][j]; B.data[j] = oneline[1][j];}
            viewer.removeShape(linesToShow.front().first);
            viewer.addLine<pcl::PointXYZRGB,pcl::PointXYZRGB>(A,B,linesToShow.front().first);
            linesToShow.pop_front();
        }
        linesToShow.clear();
    }
    viewer.spinOnce();
}

void visualizerShowCamera(const Eigen::Matrix3f& R, const Eigen::Vector3f& _t, float r, float g, float b, double s = 0.01 /*downscale factor*/, const std::string& name = "") {
    std::string name_ = name,line_name = name + "line";
    if (name.length() <= 0) {
        std::stringstream ss; ss<<"camera"<< iCamCounter++;
        name_ = ss.str();
        ss << "line";
        line_name = ss.str();
    }

    Eigen::Vector3f t = -R.transpose() * _t;

    Eigen::Vector3f vright = R.row(0).normalized() * s;
    Eigen::Vector3f vup = -R.row(1).normalized() * s;
    Eigen::Vector3f vforward = R.row(2).normalized() * s;

    Eigen::Vector3f rgb(r,g,b);

    pcl::PointCloud<pcl::PointXYZRGB> mesh_cld;
    mesh_cld.push_back(Eigen2PointXYZRGB(t,rgb));
    mesh_cld.push_back(Eigen2PointXYZRGB(t + vforward + vright/2.0 + vup/2.0,rgb));
    mesh_cld.push_back(Eigen2PointXYZRGB(t + vforward + vright/2.0 - vup/2.0,rgb));
    mesh_cld.push_back(Eigen2PointXYZRGB(t + vforward - vright/2.0 + vup/2.0,rgb));
    mesh_cld.push_back(Eigen2PointXYZRGB(t + vforward - vright/2.0 - vup/2.0,rgb));

    //TODO Mutex acquire
    pcl::PolygonMesh pm;
    pm.polygons.resize(6);
    for(int i=0;i<6;i++)
        for(int _v=0;_v<3;_v++)
            pm.polygons[i].vertices.push_back(ipolygon[i*3 + _v]);
    pcl::toROSMsg(mesh_cld,pm.cloud);
    bShowCam = true;
    cam_meshes.push_back(std::make_pair(name_,pm));
    //TODO mutex release

    linesToShow.push_back(std::make_pair(line_name,
        AsVector(Eigen2Eigen(t,rgb),Eigen2Eigen(t + vforward*3.0,rgb))
        ));
}
void visualizerShowCamera(const float R[9], const float t[3], float r, float g, float b) {
    visualizerShowCamera(Eigen::Matrix3f(R).transpose(),Eigen::Vector3f(t),r,g,b);
}
void visualizerShowCamera(const float R[9], const float t[3], float r, float g, float b, double s) {
    visualizerShowCamera(Eigen::Matrix3f(R).transpose(),Eigen::Vector3f(t),r,g,b,s);
}
void visualizerShowCamera(const cv::Matx33f& R, const cv::Vec3f& t, float r, float g, float b, double s, const std::string& name) {
    visualizerShowCamera(Eigen::Matrix<float,3,3,Eigen::RowMajor>(R.val),Eigen::Vector3f(t.val),r,g,b,s,name);
}
