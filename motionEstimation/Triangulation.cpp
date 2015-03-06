#include "Triangulation.h"

double TriangulateOpenCV(const cv::Mat K,
                         const cv::Mat distCoeff,
                         const vector<cv::Point2f>& inliersF1,
                         const vector<cv::Point2f>& inliersF2,
                         cv::Matx34f& P0,
                         cv::Matx34f& P1,
                         std::vector<cv::Point3f>& outCloud)
{
    vector<double> reproj_error;

    //triangulate Points:
    cv::Mat points1 = cv::Mat(inliersF1).reshape(1, 2);
    cv::Mat points2 = cv::Mat(inliersF2).reshape(1, 2);
    cv::Mat points3D_h = cv::Mat(1,inliersF1.size(), CV_32FC4);
    cv::triangulatePoints(P0, P1, points1, points2, points3D_h);

    //calculate reprojection
    vector<cv::Point3f> points3D;
    cv::convertPointsFromHomogeneous(points3D_h.reshape(4,1), points3D);
    cv::Mat_<double> R = (cv::Mat_<double>(3,3) << P1(0,0),P1(0,1),P1(0,2), P1(1,0),P1(1,1),P1(1,2), P1(2,0),P1(2,1),P1(2,2));
    cv::Vec3d rvec;
    cv::Rodrigues(R ,rvec);
    cv::Vec3d tvec(P1(0,3),P1(1,3),P1(2,3));
    vector<cv::Point2f> reprojected_points1;
    cv::projectPoints(points3D, rvec, tvec, K, distCoeff, reprojected_points1);

    for (unsigned int i=0; i<points3D.size(); i++) {
        outCloud.push_back(points3D[i]);
        reproj_error.push_back(norm(inliersF1[i]-reprojected_points1[i]));
    }

    cv::Scalar mse = cv::mean(reproj_error);

    //cout << "TRIANGULATE CV: Done. ("<<outCloud.size()<<"points, mean reproj err = " << mse[0] << ")"<< endl;
    return mse[0];
}


/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
cv::Mat_<double> IterativeLinearLSTriangulation(cv::Point3d point2d1_h,         //homogenous image point (u,v,1)
                                                cv::Matx34d P0,             	//camera 1 matrix
                                                cv::Point3d point2d2_h,			//homogenous image point in 2nd camera
                                                cv::Matx34d P1              	//camera 2 matrix
                                                ) {
    double wi = 1, wi1 = 1;
    cv::Mat_<double> X(4,1);
    cv::Mat_<double> X_ = LinearLSTriangulation(point2d1_h,P0,point2d2_h,P1);
    X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;
    for (int i=0; i<10; i++) { //Hartley suggests 10 iterations at most
        //recalculate weights
        double p2x = cv::Mat_<double>(cv::Mat_<double>(P0).row(2)*X)(0);
        double p2x1 = cv::Mat_<double>(cv::Mat_<double>(P1).row(2)*X)(0);

        //breaking point
        if(fabsf(wi - p2x) <= EPSILON && fabsf(wi1 - p2x1) <= EPSILON) break;

        wi = p2x;
        wi1 = p2x1;

        //reweight equations and solve
        cv::Matx43d A((point2d1_h.x*P0(2,0)-P0(0,0))/wi,		(point2d1_h.x*P0(2,1)-P0(0,1))/wi,			(point2d1_h.x*P0(2,2)-P0(0,2))/wi,
                      (point2d1_h.y*P0(2,0)-P0(1,0))/wi,		(point2d1_h.y*P0(2,1)-P0(1,1))/wi,			(point2d1_h.y*P0(2,2)-P0(1,2))/wi,
                      (point2d2_h.x*P1(2,0)-P1(0,0))/wi1,	(point2d2_h.x*P1(2,1)-P1(0,1))/wi1,		(point2d2_h.x*P1(2,2)-P1(0,2))/wi1,
                      (point2d2_h.y*P1(2,0)-P1(1,0))/wi1,	(point2d2_h.y*P1(2,1)-P1(1,1))/wi1,		(point2d2_h.y*P1(2,2)-P1(1,2))/wi1
                      );
        cv::Mat_<double> B = (cv::Mat_<double>(4,1) <<	  -(point2d1_h.x*P0(2,3)	-P0(0,3))/wi,
                              -(point2d1_h.y*P0(2,3)	-P0(1,3))/wi,
                              -(point2d2_h.x*P1(2,3)	-P1(0,3))/wi1,
                              -(point2d2_h.y*P1(2,3)	-P1(1,3))/wi1
                              );

        cv::solve(A,B,X_,cv::DECOMP_SVD);
        X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;
    }
    return X;
}


cv::Mat_<double> LinearLSTriangulation(
        cv::Point3d u,//homogenous image point (u,v,1)
        cv::Matx34d P,//camera 1 matrix
        cv::Point3d u1,//homogenous image point in 2nd camera
        cv::Matx34d P1//camera 2 matrix
        )
{
    //build A matrix
    cv::Matx43d A(u.x*P(2,0)-P(0,0),u.x*P(2,1)-P(0,1),u.x*P(2,2)-P(0,2),
              u.y*P(2,0)-P(1,0),u.y*P(2,1)-P(1,1),u.y*P(2,2)-P(1,2),
              u1.x*P1(2,0)-P1(0,0), u1.x*P1(2,1)-P1(0,1),u1.x*P1(2,2)-P1(0,2),
              u1.y*P1(2,0)-P1(1,0), u1.y*P1(2,1)-P1(1,1),u1.y*P1(2,2)-P1(1,2)
              );
    //build B vector
    cv::Matx41d B(-(u.x*P(2,3)-P(0,3)),
              -(u.y*P(2,3)-P(1,3)),
              -(u1.x*P1(2,3)-P1(0,3)),
              -(u1.y*P1(2,3)-P1(1,3)));

    //solve for X
    cv::Mat_<double> X;
    cv::solve(A,B,X,cv::DECOMP_SVD);

    // convert to homogenious 3D point
    cv::Mat_<double> XX(4, 1);
    XX(0) = X(0);
    XX(1) = X(1);
    XX(2) = X(2);
    XX(3) = 1.0;

    return XX;
}

//http://pastebin.com/UE6YW39J
double TriangulatePoints(
        const vector<cv::Point2f>& points1,
        const vector<cv::Point2f>& points2,
        const cv::Mat& K,
        const cv::Mat& Kinv,
        const cv::Matx34f& P0,
        const cv::Matx34f& P1,
        vector<cv::Point3f>& pointcloud)
{
    vector<double> reproj_error;
    cv::Mat MP1 = cv::Mat(P1);
    MP1.convertTo(MP1, CV_64F);
    cv::Mat_<double> KP1 = K * MP1;

    for (unsigned int i=0; i < points1.size(); i++) {
        //convert to normalized homogeneous coordinates
        cv::Point3f point2D1_h(points1[i].x, points1[i].y, 1.0);
        cv::Mat_<double> um = Kinv * cv::Mat_<double>(point2D1_h);
        point2D1_h = um.at<cv::Point3f>(0);

        cv::Point3f point2D2_h(points2[i].x, points2[i].y, 1.0);
        cv::Mat_<double> um1 = Kinv * cv::Mat_<double>(point2D2_h);
        point2D2_h = um1.at<cv::Point3f>(0);

        //triangulate
        //cv::Mat_<double> X = LinearLSTriangulation(point2D1_h, P0, point2D2_h, P1);
        cv::Mat_<double> X = IterativeLinearLSTriangulation(point2D1_h, P0, point2D2_h, P1);

        //calculate reprojection error
        cv::Mat_<double> xPt_img = KP1 * X;
        cv::Point2f xPt_img_(xPt_img(0) / xPt_img(2), xPt_img(1) / xPt_img(2));
        reproj_error.push_back(norm(xPt_img_ - points2[i]));

        //store 3D point
        pointcloud.push_back(cv::Point3f(X(0),X(1),X(2)));
    }

    //return mean reprojection error
    cv::Scalar me = cv::mean(reproj_error);
    return me[0];
}

