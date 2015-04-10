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



void getNewPos (cv::Mat const& currentPos, cv::Mat const& T, cv::Mat const& R, cv::Mat& newPos){
    cv::Mat temp, deltaPos;
    composeProjectionMat(T, R, temp);

    cv::Mat_<double> temp2 = (cv::Mat_<double>(1,4) << 0,0,0,1);

    if (temp.type() != temp2.type()){
        temp.convertTo(temp, temp2.type());
        std::cout << "convert" << std::endl;
    }

    cv::vconcat(temp, temp2, deltaPos);

    deltaPos.convertTo(deltaPos, CV_64F);

    newPos = currentPos * deltaPos;
}


void decomposeProjectionMat(const cv::Mat& P, cv::Mat& T, cv::Mat& R){
    R = (cv::Mat_<double>(3,3) <<
         P.at<double>(0,0),	P.at<double>(0,1),	P.at<double>(0,2),
         P.at<double>(1,0),	P.at<double>(1,1),	P.at<double>(1,2),
         P.at<double>(2,0),	P.at<double>(2,1),	P.at<double>(2,2));
    T = (cv::Mat_<double>(3,1) <<
         P.at<double>(0, 3),
         P.at<double>(1, 3),
         P.at<double>(2, 3));
}


void composeProjectionMat(const cv::Mat& T, const cv::Mat& R, cv::Mat& P){
    cv::hconcat(R, T, P);
}

void rotatePointCloud(std::vector<cv::Point3f>& cloud){
    cv::Mat R = (cv::Mat_<double>(3,3) <<
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
