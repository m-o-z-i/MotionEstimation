#define private public
#define protected public

#include "PointCloudVis.h"

pcl::visualization::PCLVisualizer viewer("MotionEstimation Viewer");


pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr orig_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

////////////////////////////////// Show Camera ////////////////////////////////////
std::deque<std::pair<std::string, pcl::PolygonMesh> >					        cam_meshes;
std::deque<std::pair<std::string, std::vector<Eigen::Matrix<float,6,1> > > >	linesToShow;
std::deque<std::pair<std::string, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> >          point_clouds;
//TODO define mutex
bool							bShowCam;
int								iCamCounter = 0;
int								iLineCounter = 0;
int								ipolygon[18] = {0,1,2,  0,3,1,  0,4,3,  0,2,4,  3,1,4,   2,4,1};

void initVisualisation(){

    cout << "INIT VISUALISATION " << endl;

    viewer.addCoordinateSystem(300,0,0,0);

    // add ground plane
    vtkSmartPointer<vtkPlaneSource> planeSource = vtkSmartPointer<vtkPlaneSource>::New ();
    planeSource->SetXResolution(40);
    planeSource->SetYResolution(40);
    planeSource->SetOrigin(-20000, 0, -20000);
    planeSource->SetPoint1(20000, 0, -20000);
    planeSource->SetPoint2(-20000, 0, 20000);

    vtkSmartPointer<vtkPolyDataMapper> planMapper = vtkSmartPointer<vtkPolyDataMapper>::New ();
    planMapper->SetInputConnection (planeSource->GetOutputPort ());

    vtkSmartPointer<vtkActor> planeActor = vtkSmartPointer<vtkActor>::New ();
    planeActor->SetMapper (planMapper);
    planeActor->GetProperty()->SetRepresentationToWireframe();
    planeActor->GetProperty()->SetColor (0.8, 0.52, .24);
    planeActor->GetProperty()->SetOpacity(0.4);

    //do not hack!!!!
    viewer.addActorToRenderer(planeActor);

    //viewer.addSphere(pcl::PointXYZ(1000,2500,5000), 50, 255, 0 ,0, "sphere");
}

void PopulatePCLPointCloud(const std::vector<cv::Point3f> &pointcloud,
                           const std::vector<cv::Vec3b>& pointcloud_RGBColor
                           )
{
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

void AddPointcloudToVisualizer(const std::vector<cv::Point3f>& pointcloud,
                               std::string name,
                               const std::vector<cv::Vec3b>& pointcloud_RGBColor) {
    //cout << "add pointcloud " << name << "  size: "  << pointcloud.size() <<  endl;

    PopulatePCLPointCloud(pointcloud,pointcloud_RGBColor);
    //SORFilter();
    pcl::copyPointCloud(*cloud,*orig_cloud);

    point_clouds.push_back(std::make_pair(name, cloud));
}

void AddLineToVisualizer(const std::vector<cv::Point3f>& pointCloud_1, const std::vector<cv::Point3f>& pointCloud_2, std::string name, const cv::Scalar &color){
    for (unsigned int i = 0; i < pointCloud_1.size(); ++i){
        pcl::PointXYZ point1(pointCloud_1[i].x, pointCloud_1[i].y, pointCloud_1[i].z);
        pcl::PointXYZ point2(pointCloud_2[i].x, pointCloud_2[i].y, pointCloud_2[i].z);
        viewer.addLine(point1, point2, color[0], color[1], color[2], name+std::to_string(i));
    }
}

void RunVisualization() {
    // draw pointclouds
    for (auto p : point_clouds) {
        viewer.addPointCloud(p.second, p.first);
    }

    point_clouds.clear();

    viewer.spinOnce();
}

void addCameraToVisualizer(const Eigen::Matrix3f& R, const Eigen::Vector3f& _t, float r, float g, float b, float s, const std::string& name) {
    std::string name_ = name,line_name = name + "line";
    if (name.length() <= 0) {
        std::stringstream ss; ss<<"camera"<< iCamCounter++;
        name_ = ss.str();
        ss << "line";
        line_name = ss.str();
    }

    Eigen::Vector3f vforward = R.col(2).normalized() * s;

    Eigen::Quaternionf RotQ(R);
    viewer.addCube(_t, RotQ, 10,10,10,name_);


    pcl::PointXYZ point1(_t(0), _t(1), _t(2));
    Eigen::Vector3f temp = _t+vforward;
    pcl::PointXYZ point2(temp(0), temp(1), temp(2));

    viewer.addLine(point1, point2, r,g,b, line_name);
}
void addCameraToVisualizer(const float R[9], const float t[3], float r, float g, float b) {
    addCameraToVisualizer(Eigen::Matrix3f(R).transpose(),Eigen::Vector3f(t),r,g,b);
}
void addCameraToVisualizer(const float R[9], const float t[3], float r, float g, float b, float s) {
    addCameraToVisualizer(Eigen::Matrix3f(R).transpose(),Eigen::Vector3f(t),r,g,b,s);
}
void addCameraToVisualizer(const cv::Vec3f& T, const cv::Matx33f& R, float r, float g, float b, float s, const std::string& name) {
    addCameraToVisualizer(Eigen::Matrix<float,3,3,Eigen::RowMajor>(R.val),Eigen::Vector3f(T.val),r,g,b,s,name);
}
void addCameraToVisualizer(const cv::Mat &T, const cv::Mat& R, float r, float g, float b, float s, const std::string& name) {
       addCameraToVisualizer(Eigen::Matrix<float,3,3,Eigen::RowMajor>(cv::Matx33f(R).val),Eigen::Vector3f(cv::Vec3f(T).val),r,g,b,s,name);
}
