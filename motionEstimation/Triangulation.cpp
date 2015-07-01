#include "Triangulation.h"

void TriangulateOpenCV(const cv::Mat& P_L,
                       const cv::Mat& P_R,
                       const vector<cv::Point2f>& points_L,
                       const vector<cv::Point2f>& points_R,
                       std::vector<cv::Point3f>& outCloud)
{
    int size = points_L.size();

    cv::Mat points3D_h(4, size, CV_32FC1);

    //triangulate Points:
    cv::triangulatePoints(P_L, P_R, points_L, points_R, points3D_h);

    cv::convertPointsFromHomogeneous(cv::Mat(points3D_h.t()).reshape(4,1), outCloud);

}

/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
cv::Mat_<float> IterativeLinearLSTriangulation(cv::Point3f point2d1_h,         //homogenous image point (u,v,1)
                                                cv::Matx34f P0,             	//camera 1 matrix
                                                cv::Point3f point2d2_h,			//homogenous image point in 2nd camera
                                                cv::Matx34f P1              	//camera 2 matrix
                                                ) {
    float wi = 1, wi1 = 1;
    cv::Mat_<float> X(4,1);
    cv::Mat_<float> X_ = LinearLSTriangulation(point2d1_h,P0,point2d2_h,P1);
    X(0) = X_(0);
    X(1) = X_(1);
    X(2) = X_(2);
    X(3) = 1.0;

    //doesn't make sense!?
    for (int i=0; i<10; i++) { //Hartley suggests 10 iterations at most

        //recalculate weights
        float p2x = cv::Mat_<float>(cv::Mat_<float>(P0).row(2)*X)(0);
        float p2x1 = cv::Mat_<float>(cv::Mat_<float>(P1).row(2)*X)(0);

        //breaking point
        if(fabsf(wi - p2x) <= EPSILON && fabsf(wi1 - p2x1) <= EPSILON){
            break;
        }

        wi = p2x;
        wi1 = p2x1;

        //reweight equations and solve
        cv::Matx43f A(
                    (point2d1_h.x*P0(2,0)-P0(0,0))/wi,	(point2d1_h.x*P0(2,1)-P0(0,1))/wi,	(point2d1_h.x*P0(2,2)-P0(0,2))/wi,
                    (point2d1_h.y*P0(2,0)-P0(1,0))/wi,	(point2d1_h.y*P0(2,1)-P0(1,1))/wi,	(point2d1_h.y*P0(2,2)-P0(1,2))/wi,
                    (point2d2_h.x*P1(2,0)-P1(0,0))/wi1,	(point2d2_h.x*P1(2,1)-P1(0,1))/wi1,	(point2d2_h.x*P1(2,2)-P1(0,2))/wi1,
                    (point2d2_h.y*P1(2,0)-P1(1,0))/wi1,	(point2d2_h.y*P1(2,1)-P1(1,1))/wi1,	(point2d2_h.y*P1(2,2)-P1(1,2))/wi1
                    );

        cv::Mat_<float> B = (cv::Mat_<float>(4,1) <<
                              -(point2d1_h.x*P0(2,3)	-P0(0,3))/wi,
                              -(point2d1_h.y*P0(2,3)	-P0(1,3))/wi,
                              -(point2d2_h.x*P1(2,3)	-P1(0,3))/wi1,
                              -(point2d2_h.y*P1(2,3)	-P1(1,3))/wi1
                              );

        cv::solve(A,B,X_,cv::DECOMP_SVD);
        X(0) = X_(0);
        X(1) = X_(1);
        X(2) = X_(2);
        X(3) = 1.0;
    }


    return X;
}

/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
cv::Mat_<float> LinearLSTriangulation(
        cv::Point3f u,//homogenous image point (u,v,1)
        cv::Matx34f P,//camera 1 matrix
        cv::Point3f u1,//homogenous image point in 2nd camera
        cv::Matx34f P1//camera 2 matrix
        )
{
    //build matrix A for homogenous equation system Ax = 0
    //assume X = (x,y,z,1), for Linear-LS method
    //which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
    cv::Matx43f A(
                u.x*P(2,0)-P(0,0),    u.x*P(2,1)-P(0,1),      u.x*P(2,2)-P(0,2),
                u.y*P(2,0)-P(1,0),    u.y*P(2,1)-P(1,1),      u.y*P(2,2)-P(1,2),
                u1.x*P1(2,0)-P1(0,0), u1.x*P1(2,1)-P1(0,1),   u1.x*P1(2,2)-P1(0,2),
                u1.y*P1(2,0)-P1(1,0), u1.y*P1(2,1)-P1(1,1),   u1.y*P1(2,2)-P1(1,2)
                );

    //build B vector
    cv::Mat_<float> B = (cv::Mat_<float>(4,1) <<
                          -(u.x*P(2,3)      -P(0,3)),
                          -(u.y*P(2,3)      -P(1,3)),
                          -(u1.x*P1(2,3)    -P1(0,3)),
                          -(u1.y*P1(2,3)    -P1(1,3))
                          );

    //solve for X
    cv::Mat_<float> X;
    cv::solve(A,B,X,cv::DECOMP_SVD);

    return X;
}

//http://pastebin.com/UE6YW39J
void TriangulatePointsHZ(const cv::Mat& P_L, const cv::Mat& P_R, //normalized PK = K * P
                         const vector<cv::Point2f>& points1,
                         const vector<cv::Point2f>& points2,
                         int numberOfTriangulations,
                         vector<cv::Point3f>& pointcloud)
{
    // if parameter is 0 triangulate all points
    if (0 == numberOfTriangulations) {
        numberOfTriangulations = points1.size();
    }

    int interval = (points1.size() / numberOfTriangulations);
    if (1 > interval) {
        interval = 1;
    }

    pointcloud.clear();

    vector<cv::Point3f> points1_h, points2_h;
    cv::convertPointsToHomogeneous(points1, points1_h);
    cv::convertPointsToHomogeneous(points2, points2_h);

    int index = 0;
    for (unsigned int i=0; i < numberOfTriangulations; ++i ){
        cv::Mat_<float> X = LinearLSTriangulation(points1_h[index],P_L,points2_h[index],P_R);
        pointcloud.push_back(cv::Point3f(X(0),X(1),X(2)));
        index+=interval;
    }
}

void TriangulatePointsWithInlier(
        const cv::Matx34f& P0,
        const cv::Matx34f& P1,
        const vector<cv::Point2f>& points1,
        const vector<cv::Point2f>& points2,
        int numberOfTriangulations,
        vector<cv::Point3f>& pointcloud,
        vector<cv::Point2f>& inlier1,
        vector<cv::Point2f>& inlier2
        )
{
    // if parameter is 0 triangulate all points
    if (0 == numberOfTriangulations) {
        numberOfTriangulations = points1.size();
    }
    pointcloud.clear();

    vector<cv::Point3f> points1_h, points2_h;
    cv::convertPointsToHomogeneous(points1, points1_h);
    cv::convertPointsToHomogeneous(points2, points2_h);

    for (unsigned int i=0; i < numberOfTriangulations; ++i ){
        cv::Mat_<float> X = IterativeLinearLSTriangulation(points1_h[i],P0,points2_h[i],P1);
        if (0 < X(2)){
            pointcloud.push_back(cv::Point3f(X(0),X(1),X(2)));
            inlier1.push_back(points1[i]);
            inlier2.push_back(points2[i]);
        } else {
            pointcloud.push_back(cv::Point3f(0,0,0));
            inlier1.push_back(cv::Point2f(0,0));
            inlier2.push_back(cv::Point2f(0,0));
        }
    }
}

void triangulate(const cv::Mat& P0, const cv::Mat& P1, const vector<cv::Point2f>& x0, const vector<cv::Point2f>& x1, vector<cv::Point3f>& result3D) {
    assert(x0.size() == x1.size());
    result3D.clear();

    for(uint i = 0; i < x0.size(); i++) {
        //set up a system of linear equations from x = PX and x' = P'X
        cv::Mat A(4, 4, CV_32FC1);
        A.row(0) = x0[i].x * P0.row(2) - P0.row(0);
        A.row(1) = x0[i].y * P0.row(2) - P0.row(1);
        A.row(2) = x1[i].x * P1.row(2) - P1.row(0);
        A.row(3) = x1[i].y * P1.row(2) - P1.row(1);
        //Utils::printMatrix(A, "A:");

        //normalize each row of A with its L2 norm, i.e. |row| = sqrt(sum_j(row[j]^2)) to improve condition of the system
        for (int i = 0; i < A.rows; i++) {
            float dsquared = 0;
            for(int j = 0; j < 4; j++) {
                dsquared = dsquared + pow(A.at<float>(i, j), 2);
            }
            A.row(i) = A.row(i) * (1 / sqrt(dsquared));
        }

        float detA = cv::determinant(A);
        //cout << setprecision(3) << "det(A): " << detA << endl;
        if(detA < 0.0) {
            //workaround SVD ambiguity if det < 0
            A = A * -1.0;
        }

        //solve A x = 0 using Singular Value Decomposition
        cv::SVD decomposition(A);

        //homogeneous least-square solution corresponds to least singular vector of A, that is the last column of V or last row of V^T
        //i.e. [x,y,z,w] = V^T.row(3)
        float x = static_cast<float>(decomposition.vt.at<float>(3, 0));
        float y = static_cast<float>(decomposition.vt.at<float>(3, 1));
        float z = static_cast<float>(decomposition.vt.at<float>(3, 2));
        float w = static_cast<float>(decomposition.vt.at<float>(3, 3));
        //convert homogeneous to cartesian coordinates
        result3D.push_back(cv::Point3f(x/w, y/w, z/w));

        //cout << "2D Coordinates x: " << x0[i].x << ", " << x0[i].y << "  and  x': " << x1[i].x << ", " << x1[i].y << endl;
        //cout << "3D Coordinate  X: " << x/w << ", " << y/w << ", " << z/w <<  endl;
        //cout << "Homogeneous 3D Coordinate X : " << x << ", " << y << ", " << z << ", " << w << endl;
    }
}

void computeReprojectionError(const cv::Mat& P,
                              const vector<cv::Point2f>& points,
                              const vector<cv::Point3f>& worldCoordinates,
                              vector<cv::Point3f>& pReprojected,
                              vector<cv::Point2f>& reprojectionErrors,
                              cv::Point2f& avgReprojectionError)
{
    assert(points.size() == worldCoordinates.size());

    //for all points...
    for(uint i = 0; i < points.size(); i++) {

        //build homogeneous coordinate for projection
        cv::Mat WorldCoordinate_h = cv::Mat(4, 1, CV_32FC1);
        WorldCoordinate_h.at<float>(0,0) = worldCoordinates[i].x;
        WorldCoordinate_h.at<float>(1,0) = worldCoordinates[i].y;
        WorldCoordinate_h.at<float>(2,0) = worldCoordinates[i].z;
        WorldCoordinate_h.at<float>(3,0) = 1.0;

        //perform simple reprojection by multiplication with projection matrix
        cv::Mat pReprojected_h = P * WorldCoordinate_h; //homogeneous image coordinates 3x1

        //convert reprojected image point to carthesian coordinates
        float w = static_cast<float>(pReprojected_h.at<float>(2,0));
        float x_r = static_cast<float>(pReprojected_h.at<float>(0,0) / w); //x = x/w
        float y_r = static_cast<float>(pReprojected_h.at<float>(1,0) / w); //y = y/w

        pReprojected.push_back(cv::Point3f(x_r, y_r, w)); //reprojected cartesian image coordinate with depth value

        //calculate actual reprojection error
        float deltaX = (float)fabs(points[i].x - x_r);
        float deltaY = (float)fabs(points[i].y - y_r);
        reprojectionErrors.push_back(cv::Point2f(deltaX, deltaY));
    }

    //average reprojection error
    avgReprojectionError.x = avgReprojectionError.y = 0.0;
    for(uint i = 0; i < reprojectionErrors.size(); i++) {
        avgReprojectionError.x += reprojectionErrors[i].x;
        avgReprojectionError.y += reprojectionErrors[i].y;
    }
    avgReprojectionError.x /= reprojectionErrors.size();
    avgReprojectionError.y /= reprojectionErrors.size();
}

float calculateReprojectionErrorOpenCV(const cv::Mat& P,
                                        const cv::Mat& K,
                                        const cv::Mat distCoeff,
                                        const vector<cv::Point2f>& points2D,
                                        const std::vector<cv::Point3f>& points3D)
{
    vector<float> reproj_error;

    cv::Matx34f P_(P);
    cv::Mat_<float> R = (cv::Mat_<float>(3,3) <<
                          P_(0,0),P_(0,1),P_(0,2),
                          P_(1,0),P_(1,1),P_(1,2),
                          P_(2,0),P_(2,1),P_(2,2));
    cv::Mat_<float> T = (cv::Mat_<float>(1,3) << P_(0,3),P_(1,3),P_(2,3));

    //calculate reprojection
    cv::Vec3d rvec;
    cv::Rodrigues(R ,rvec);
    cv::Vec3d tvec(T);

    vector<cv::Point2f> reprojected_points2D;
    vector<float > distCoeffVec; //just use empty vector.. images are allready undistorted..
    cv::projectPoints(points3D, rvec, tvec, K, distCoeff, reprojected_points2D);

    for (unsigned int i=0; i<points3D.size(); i++) {
        reproj_error.push_back(cv::norm(points2D[i]-reprojected_points2D[i]));
    }

    cv::Scalar mse = cv::mean(reproj_error);

    return mse[0];
}

float calculateReprojectionErrorHZ(const cv::Mat& P,
                                    const vector<cv::Point2f>& points2D,
                                    const std::vector<cv::Point3f>& points3D)
{
    vector<float> reproj_error;
    //    cv::Mat P_(P);
    //    P_.convertTo(P_, CV_32F);
    //    cv::Mat_<float> KP = K * P_;

    for (unsigned int i=0; i < points3D.size(); i++) {
        // convert to homogenious 3D point
        cv::Mat_<float> Point3f_h(4, 1);
        Point3f_h(0) = points3D[i].x;
        Point3f_h(1) = points3D[i].y;
        Point3f_h(2) = points3D[i].z;
        Point3f_h(3) = 1.0;

        // calculate reprojection error ((( KP * points3D[i] ???)))
        cv::Mat_<float> reprojectedPoint_h = P * Point3f_h;

        // convert reprojected image point to carthesian coordinates
        cv::Point2f reprojectedPoint(reprojectedPoint_h(0) / reprojectedPoint_h(2), reprojectedPoint_h(1) / reprojectedPoint_h(2));

        reproj_error.push_back(cv::norm(points2D[i] - reprojectedPoint));
    }

    //return mean reprojection error
    cv::Scalar me = cv::mean(reproj_error);
    return me[0];
}
