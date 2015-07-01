#include "FindCameraMatrices.h"
#include "Triangulation.h"


bool getRightProjectionMat( cv::Mat& E,
                            cv::Mat& P1,
                            const cv::Mat& K,
                            const vector<cv::Point2f>& points2D_1,
                            const vector<cv::Point2f>& points2D_2,
                            std::vector<cv::Point3f>& outCloud)
{
    // no rotation or translation for the left projection matrix
    //projection matrix of first camera P0 = K[I|0]
    cv::Mat P0 = (cv::Mat_<float>(3,4) <<
                  1.0, 0.0, 0.0, 0.0,
                  0.0, 1.0, 0.0, 0.0,
                  0.0, 0.0, 1.0, 0.0 );

    //according to http://en.wikipedia.org/wiki/Essential_matrix#Properties_of_the_essential_matrix
    if(fabsf(determinant(E)) > 5) { // > 1e-03
        cout << "det(E) != 0 : " << determinant(E) << "\n";
        return false;
    }

    cv::Mat_<float> R1(3,3);
    cv::Mat_<float> R2(3,3);
    cv::Mat_<float> t1(1,3);
    cv::Mat_<float> t2(1,3);

    //decompose E to P1 , HZ (9.19)
    {
        // validation of E
        bool ValidationOfE = DecomposeEtoRandT(E,R1,R2,t1,t2);     // extract cameras [R|t]
        if (!ValidationOfE) return false;
        if(determinant(R1)+1.0 < 1e-03 || determinant(R2)+1.0 < 1e-03) {
            //according to http://en.wikipedia.org/wiki/Essential_matrix#Showing_that_it_is_valid
            cout << "det(R) == -1 ["<<determinant(R1)<<"]: flip E's sign" << endl;
            E = -E;
            DecomposeEtoRandT(E,R1,R2,t1,t2);
        }
        if (!CheckCoherentRotation(R1) && !CheckCoherentRotation(R2)) {
            cout << "det(R) != +-1.0, this is not a rotation matrix" << endl;
            return false;
        }

        std::vector<cv::Mat_<float>> Rotations{R1,R2};
        std::vector<cv::Mat_<float>> Translations{t1,t2};

        int counter = 0;
        std::vector<cv::Point3f> pcloud;
        bool foundPerspectiveMatrix = false;

        // find right solution of 4 possible translations and rotations
        for (unsigned int i = 0; i < 2; ++i) {

            if (foundPerspectiveMatrix){
                break;
            }

            cv::Mat_<float> R = Rotations[i];
            if (!CheckCoherentRotation(R)) {
                cout << "resulting rotation R is not coherent\n";
                counter += 2;
                continue;
            }

            for (unsigned int j = 0; j < 2; ++j) {
                pcloud.clear();
                cv::Mat_<float> T = Translations[j];
                //cout << "\n************ Testing P"<< i<<j<< " **************" << endl;

                //projection matrix of second camera: P1  = K[R|t]
                composeProjectionMat(T, R, P1);

                // calibrate projection Mat
                cv::Mat PK_0 = K * P0;
                cv::Mat PK_1 = K * P1;

                //triangulate from Richard Hartley and Andrew Zisserman
                TriangulatePointsHZ(PK_0, PK_1, points2D_1, points2D_2, 20, pcloud);

                float reproj_error_L = calculateReprojectionErrorHZ(PK_0, points2D_1, pcloud);
                float reproj_error_R = calculateReprojectionErrorHZ(PK_1, points2D_2, pcloud);

                //check if pointa are triangulated --in front-- of both cameras. If yes break loop
                if (positionCheck(P0, pcloud) && positionCheck(P1, pcloud)) {
                    //cout << "############## use this perspective Matrix " << "P"<< i << j << "################" << endl;
                    //cout << "HZ: reprojection ERROR:  left:  " <<  reproj_error_L << "  right  " << reproj_error_R << endl;
                    foundPerspectiveMatrix = true;
                    break;
                }
                ++counter;
            }
        }

        if (4 == counter) {
            cout << "NO MOVEMENT: Can't find any right perspective Mat" << endl;
            return false;
        }

        for (unsigned int i=0; i<pcloud.size(); i++) {
            outCloud.push_back(pcloud[i]);
        }
    }

    return true;
}

bool positionCheck(const cv::Matx34f& P, const std::vector<cv::Point3f>& points3D) {
    vector<cv::Point3f> pcloud_pt3d_projected(points3D.size());

    cv::Matx44f P4x4 = cv::Matx44f::eye();
    for(int i=0;i<12;i++) P4x4.val[i] = P.val[i];

    cv::perspectiveTransform(points3D, pcloud_pt3d_projected, P4x4);

    // status of 3d points..infront: >0 or behind: <0
    vector<uchar> status;
    for (unsigned int i=0; i<points3D.size(); i++) {
        status.push_back((pcloud_pt3d_projected[i].z > 0) ? 1 : 0);
    }
    int count = cv::countNonZero(status);

    float percentage = ((float)count / (float)points3D.size());
    std::cout << count << "/" << points3D.size() << " = " << percentage*100.0 << "% are in front of camera" << std::endl;
    if(percentage < 0.55){
        //less than 55% of the points are in front of the camera
        return false;
    }
    return true;
}

bool DecomposeEtoRandT(const cv::Mat& E,
                       cv::Mat_<float>& R1,
                       cv::Mat_<float>& R2,
                       cv::Mat_<float>& t1,
                       cv::Mat_<float>& t2)
{
#if 0
    // decompose the essential matrix to P', HZ 9.19
    cv::SVD svd(E, cv::SVD::MODIFY_A);
    cv::Mat svd_u = svd.u;
    cv::Mat svd_vt = svd.vt;

    // HZ 9.13
    cv::Matx33d w(0, -1, 0,
                  1, 0, 0,
                  0, 0, 1);

    cv::Mat_<float> R = svd_u * cv::Mat(w) * svd_vt; // HZ 9.19
    cv::Mat_<float> T = svd_u.col(2); // u3

    if (!CheckCoherentRotation(R)) {
        std::cout << "resulting rotation is not coherent" << std::endl;
        return 0;
    }

    // P' the second camera matrix, in the form of R|t
    // (rotation & translation)
    cv::Matx34f P1;
    P1 = cv::Matx34f(R(0, 0), R(0, 1), R(0, 2), T(0),
                     R(1, 0), R(1, 1), R(1, 2), T(1),
                     R(2, 0), R(2, 1), R(2, 2), T(2));
#endif

    //Using HZ E decomposition
    cv::SVD svd(E, cv::SVD::FULL_UV); //MODIFY_A
    cv::Mat svd_u = svd.u;  // U
    cv::Mat svd_w = svd.w; // D
    cv::Mat svd_vt = svd.vt; // V transpose

    //check if first and second singular values are the same (as they should be)
    float singular_values_ratio = fabsf(svd_w.at<float>(0) / svd_w.at<float>(1));
    if((singular_values_ratio < 0.8 || singular_values_ratio > 1.2) && svd_w.at<float>(2) < 1e-04){
        std::cout << "#####################################################" << std::endl;
        std::cout << "singular values are too far apart... no rot and trans for this frame" << std::endl;
        return false;
    }

    //HZ 9.13
    cv::Matx33f W(0,-1, 0,
                  1, 0, 0,
                  0, 0, 1);

    cv::Matx33f Wt(0, 1, 0,
                  -1, 0, 0,
                   0, 0, 1);

    R1 = svd_u * cv::Mat(W) * svd_vt; //HZ 9.19
    R2 = svd_u * cv::Mat(Wt) * svd_vt; //HZ 9.19
    t1 = svd_u.col(2); //u3
    t2 = -svd_u.col(2); //u3

    return true;
}

/*One more thing we can think of adding to our method is error checking.
 * Many a times the calculation of the fundamental matrix from the point matching is erroneous,
 * and this affects the camera matrices.
 * Continuing triangulation with faulty camera matrices is pointless.
 * We can install a check to see if the rotation element is a valid rotation matrix.
 * Keeping in mind that rotation matrices must have a determinant of 1 (or -1),
 * we can simply do the following:
 */
bool CheckCoherentRotation(cv::Mat const& R) {
    if(fabsf(determinant(R))-1.0 > 1e-03) {
        return false;
    }
    return true;
}

bool getFundamentalMatrix(vector<cv::Point2f>  const& points1, vector<cv::Point2f> const& points2, vector<cv::Point2f> *inliers1, vector<cv::Point2f> *inliers2, cv::Mat& F) {
    // Compute F matrix using RANSAC
    if(points1.size() != points2.size() || 0 == points1.size()){
        return false;
    }

    //vector<cv::Point2f> p1;
    //vector<cv::Point2f> p2;

    std::vector<uchar> inliers_fundamental(points1.size(),0);
    F = cv::findFundamentalMat(
                cv::Mat(points1), cv::Mat(points2),   // matching points
                inliers_fundamental,                             // match status (inlier ou outlier)
                cv::FM_RANSAC,                                   // RANSAC method
                5.,                                              // distance to epipolar line
                .01);                                            // confidence probability

    if(countNonZero(F) < 1) {
        //cout << "can't find F" << endl;
        return false;
    }

    //check x' * F * x = 0 ??
    vector<cv::Point3f> homogenouse1;
    vector<cv::Point3f> homogenouse2;
    cv::convertPointsToHomogeneous(points1, homogenouse1);
    cv::convertPointsToHomogeneous(points2, homogenouse2);

    F.convertTo(F, CV_32F);

    //get Inlier
    for(unsigned i = 0; i<points1.size(); ++i){
        if (inliers_fundamental[i] == 1) {
            cv::Mat point1 (homogenouse1[i]);
            cv::Mat point2 (homogenouse2[i]);

            cv::Mat calc = point2.t() * F * point1;

            // get value of calculation point2.t() * F * point1 should be 0
            if (calc.at<float>(0,0) <= 0.1){
                inliers1->push_back(points1[i]);
                inliers2->push_back(points2[i]);
            } else {
                inliers1->push_back(cv::Point2f(0,0));
                inliers2->push_back(cv::Point2f(0,0));
            }

        } else {
            inliers1->push_back(cv::Point2f(0,0));
            inliers2->push_back(cv::Point2f(0,0));
        }
    }

   return true;
}


//-----------------------------------------------------------------------------
void loadIntrinsic(string path, cv::Mat& K_L, cv::Mat& K_R, cv::Mat& distCoeff_L, cv::Mat& distCoeff_R) {
    //-----------------------------------------------------------------------------
    cv::FileStorage fs(path + "calibration/intrinsic.yml", cv::FileStorage::READ);
    fs["cameraMatrixLeft"] >> K_L;
    fs["cameraMatrixRight"] >> K_R;
    fs["distCoeffsLeft"] >> distCoeff_L;
    fs["distCoeffsRight"] >> distCoeff_R;
    fs.release();
}

//-----------------------------------------------------------------------------
void loadExtrinsic(string path, cv::Mat& R, cv::Mat& T, cv::Mat& E, cv::Mat& F ) {
    //-----------------------------------------------------------------------------
    cv::FileStorage fs(path + "calibration/extrinsic.yml", cv::FileStorage::READ);
    fs["R"] >> R;
    fs["T"] >> T;
    fs["E"] >> E;
    fs["F"] >> F;
    fs.release();
}
struct Comp{
    Comp( const std::vector<cv::Point3f>& v ) : _v(v) {}
    bool operator ()(cv::Point3f a, cv::Point3f b) { return cv::norm(a) < cv::norm(b); }
    const std::vector<cv::Point3f>& _v;
};

void getScaleFactor(const cv::Mat& P0, const cv::Mat& P_LR, const cv::Mat& P_L, const cv::Mat& P_R,
                    const std::vector<cv::Point2f>& points_L1, const std::vector<cv::Point2f>& points_R1,
                    const std::vector<cv::Point2f>& points_L2, const std::vector<cv::Point2f>& points_R2,
                    float& u, float& v, std::vector<cv::Point3f>& pCloud, std::vector<cv::Point3f>& nearestPoints)
{
    std::vector<cv::Point3f> X_L, X_R;
    TriangulatePointsHZ(P0, P_LR, points_L1, points_R1, 0, pCloud);
    TriangulatePointsHZ(P0, P_L , points_L1, points_L2, 0, X_L);
    TriangulatePointsHZ(P0, P_R , points_R1, points_R2, 0, X_R);

    // find 5 nearest points:
    std::vector<cv::Point3f> temp(pCloud);
    partial_sort( temp.begin(), temp.begin()+5, temp.end(), Comp(temp) );

    std::vector<cv::Point3f> X_5, X_L_5, X_R_5;
    std::vector<cv::Point3f>::iterator it;
    for(unsigned int i = 0; i < 5; ++i){
        it = find(pCloud.begin(), pCloud.end(), temp[i]);
        int index = it - pCloud.begin();

        X_5.push_back(pCloud[index]);
        nearestPoints.push_back(pCloud[index]);
        X_L_5.push_back(X_L[index]);
        X_R_5.push_back(X_R[index]);
    }


    float sum_L = 0;
    float sum_R = 0;
    for (unsigned int i = 0; i < X_5.size(); ++i) {
        sum_L += ((cv::norm(X_5[i])*1.0) / cv::norm(X_L[i])*1.0);
        sum_R += ((cv::norm(X_5[i])*1.0) / cv::norm(X_R[i])*1.0);
    }

    u = 1.0/X_5.size() * sum_L;
    v = 1.0/X_5.size() * sum_R;
}

void getScaleFactorLeft(const cv::Mat& P0, const cv::Mat& P_LR, const cv::Mat& P_L,
                    const std::vector<cv::Point2f>& points_L1, const std::vector<cv::Point2f>& points_R1,
                    const std::vector<cv::Point2f>& points_L2,
                    float& u)
{
    std::vector<cv::Point3f> X, X_L;
    TriangulatePointsHZ(P0, P_LR, points_L1, points_R1, 0, X);
    TriangulatePointsHZ(P0, P_L , points_L1, points_L2, 0, X_L);

    float sum_L = 0;
    for (unsigned int i = 0; i < X.size(); ++i) {
        sum_L += ((cv::norm(X[i])*1.0) / cv::norm(X_L[i])*1.0);
    }

    u = 1.0/X.size() * sum_L;
}

void getScaleFactorRight(const cv::Mat& P0, const cv::Mat& P_LR, const cv::Mat& P_R,
                    const std::vector<cv::Point2f>& points_L1, const std::vector<cv::Point2f>& points_R1,
                    const std::vector<cv::Point2f>& points_R2,
                    float& u)
{
    std::vector<cv::Point3f> X, X_R;
    TriangulatePointsHZ(P0, P_LR, points_L1, points_R1, 0, X);
    TriangulatePointsHZ(P0, P_R , points_R1, points_R2, 0, X_R);

    float sum_R = 0;
    for (unsigned int i = 0; i < X.size(); ++i) {
        sum_R += ((cv::norm(X[i])*1.0) / cv::norm(X_R[i])*1.0);
    }

    u = 1.0/X.size() * sum_R;
}

void getScaleFactor2(const cv::Mat& T_LR, const cv::Mat& R_LR, const cv::Mat& T_L, const cv::Mat& R_L, const cv::Mat& T_R,  float& u, float& v) {
    cv::Mat A(3, 2, CV_32F);
    cv::Mat B(3, 1, CV_32F);
    cv::Mat x(2, 1, CV_32F);

    cv::hconcat(T_L, -(R_LR*T_R), A);
    B = T_LR -(R_L*T_LR);

//    cout << "\n\n\n  ########### Matrizen ############### \n A: \n "<< A << endl << endl;
//    cout << "\n B: \n "<< B << endl << endl;
//    cout << "\n x: \n "<< x << endl << endl;

    //solve Ax = B
    cv::solve(A, B, x, cv::DECOMP_SVD);
    u = x.at<float>(0,0);
    v = x.at<float>(1,0);
}

