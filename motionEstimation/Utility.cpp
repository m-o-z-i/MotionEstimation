#include "Utility.h"

int getFiles (std::string const& dir, std::vector<std::string> &files)
{
    DIR *dp;
    struct dirent *dirp;

    //Unable to open dir
    if((dp  = opendir(dir.c_str())) == NULL)
    {
        std::cout << "Error(" << errno << ") opening " << dir << std::endl;
        return errno;
    }

    //read files and push them to vector
    while ((dirp = readdir(dp)) != NULL)
    {
        std::string name = std::string(dirp->d_name);
        //discard . and .. from list
        if(name != "." && name != "..")
        {
            files.push_back(std::string(dirp->d_name));
        }
    }

    closedir(dp);
    std::sort(files.begin(), files.end());

    return 0;
}

void getAbsPos (cv::Mat const& currentPos, cv::Mat const& T, cv::Mat const& R, cv::Mat& newPos){
    cv::Mat temp, deltaPos;
    composeProjectionMat(T, R, temp);

    cv::Mat_<float> temp2 = (cv::Mat_<float>(1,4) << 0,0,0,1);

    if (temp.type() != temp2.type()){
        temp.convertTo(temp, temp2.type());
    }

    cv::vconcat(temp, temp2, deltaPos);

    deltaPos.convertTo(deltaPos, CV_32F);

    newPos = currentPos * deltaPos;
}

void getNewTrans3D (cv::Mat const& T, cv::Mat const& R, cv::Mat& position){
    position = -R.t() * T;
}


void decomposeProjectionMat(const cv::Mat& P, cv::Mat& T, cv::Mat& R){
    R = (cv::Mat_<float>(3,3) <<
         P.at<float>(0,0),	P.at<float>(0,1),	P.at<float>(0,2),
         P.at<float>(1,0),	P.at<float>(1,1),	P.at<float>(1,2),
         P.at<float>(2,0),	P.at<float>(2,1),	P.at<float>(2,2));
    T = (cv::Mat_<float>(3,1) <<
         P.at<float>(0, 3),
         P.at<float>(1, 3),
         P.at<float>(2, 3));
}


void composeProjectionMat(const cv::Mat& T, const cv::Mat& R, cv::Mat& P){
    cv::hconcat(R, T, P);
}

void decomposeRotMat(const cv::Mat& R, float& x, float& y, float& z){
    x = atan2(R.at<float>(2,1), R.at<float>(2,2)) * (180 / M_PI);
    y = atan2(R.at<float>(2,0), sqrt(pow(R.at<float>(2,1),2)+pow(R.at<float>(2,2),2))) * (180 / M_PI);
    z = atan2(R.at<float>(1,0), R.at<float>(0,0)) * (180 / M_PI);
}

void rotatePointCloud(std::vector<cv::Point3f>& cloud){
    cv::Mat R = (cv::Mat_<float>(3,3) <<
                -1, 0, 0,
                 0,-1, 0,
                 0, 0, 1);
    for (auto &i : cloud){
        cv::Mat point(i);
        point.convertTo(point, R.type());
        cv::Mat newPoint = R * point;
        i = cv::Point3f(newPoint);
    }
}

void rotatePointCloud(std::vector<cv::Point3f>& cloud, const cv::Mat P){
    cv::Mat R, T;
    decomposeProjectionMat(P, T, R);

    for (auto &i : cloud){
        cv::Mat point(i);
        point.convertTo(point, R.type());

        cv::Mat newPoint = R * point + T;
        i = cv::Point3f(newPoint);
    }
}

void rotateRandT(cv::Mat& Trans, cv::Mat& Rot){
    // Rotate R and T Mat
    cv::Mat R = (cv::Mat_<float>(3,3) <<
                -1, 0,  0,
                 0, 1,  0,
                 0, 0, -1);

    Trans = R * Trans;
    Rot = R * Rot;
}

void KeyPointsToPoints(const std::vector<cv::KeyPoint>& kps, std::vector<cv::Point2f>& ps) {
    ps.clear();
    for (unsigned int i=0; i<kps.size(); i++) ps.push_back(kps[i].pt);
}

void PointsToKeyPoints(const std::vector<cv::Point2f>& ps, std::vector<cv::KeyPoint>& kps) {
    kps.clear();
    for (unsigned int i=0; i<ps.size(); i++) kps.push_back(cv::KeyPoint(ps[i],1.0f));
}

bool calcCoordinate(cv::Mat_<float> &toReturn,cv::Mat const& Q, cv::Mat const& disparityMap,int x,int y)
{
    double d = static_cast<float>(disparityMap.at<short>(y,x));
    d/=16.0;
    if(d > 0)
    {
      toReturn(0)=x;
      toReturn(1)=y;
      toReturn(2)=d;
      toReturn(3)=1;

      toReturn = Q*toReturn.t();
      toReturn/=toReturn(3);
      return true;
    }
    else
    {
      return false;
    }
}


