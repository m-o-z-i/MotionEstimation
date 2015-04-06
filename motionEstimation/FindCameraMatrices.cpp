#include "FindCameraMatrices.h"
#include "Triangulation.h"


bool getRightProjectionMat( cv::Mat& E,
                            cv::Mat& P1,
                            const vector<cv::Point2f>& normPoints2D_L,
                            const vector<cv::Point2f>& normPoints2D_R,
                            std::vector<cv::Point3f>& outCloud)
{
    // no rotation or translation for the left projection matrix
    //projection matrix of first camera P0 = K[I|0]
    cv::Mat P0 = (cv::Mat_<double>(3,4) <<
                  1.0, 0.0, 0.0, 0.0,
                  0.0, 1.0, 0.0, 0.0,
                  0.0, 0.0, 1.0, 0.0 );

    //according to http://en.wikipedia.org/wiki/Essential_matrix#Properties_of_the_essential_matrix
    if(fabsf(determinant(E)) > 1e-03) {
        cout << "det(E) != 0 : " << determinant(E) << "\n";
        return false;
    }

    cv::Mat_<double> R1(3,3);
    cv::Mat_<double> R2(3,3);
    cv::Mat_<double> t1(1,3);
    cv::Mat_<double> t2(1,3);

    //decompose E to P1 , HZ (9.19)
    {
        // validation of E
        bool ValidationOfE = DecomposeEtoRandT(E,R1,R2,t1,t2);     // extract cameras [R|t]
        if (!ValidationOfE) return false;
        if(determinant(R1)+1.0 < 1e-05) {
            //according to http://en.wikipedia.org/wiki/Essential_matrix#Showing_that_it_is_valid
            cout << "det(R) == -1 ["<<determinant(R1)<<"]: flip E's sign" << endl;
            E = -E;
            DecomposeEtoRandT(E,R1,R2,t1,t2);
        }
        if (!CheckCoherentRotation(R1) && !CheckCoherentRotation(R2)) {
            cout << "resulting rotations are not coherent\n";
            return false;
        }

        std::vector<cv::Mat_<double>> Rotations{R1,R2};
        std::vector<cv::Mat_<double>> Translations{t1,t2};

        int counter = 0;
        std::vector<cv::Point3f> pcloud, pcloud1, worldCoordinates;
        bool foundPerspectiveMatrix = false;

        // find right solution of 4 possible translations and rotations
        for (unsigned int i = 0; i < 2; ++i) {

            if (foundPerspectiveMatrix){
                break;
            }

            cv::Mat_<double> R = Rotations[i];
            if (!CheckCoherentRotation(R)) {
                cout << "resulting rotation R is not coherent\n";
                counter += 2;
                continue;
            }

            for (unsigned int j = 0; j < 2; ++j) {
                pcloud.clear(); pcloud1.clear(); worldCoordinates.clear();
                cv::Mat_<double> T = Translations[j];
                //cout << "\n************ Testing P"<< i<<j<< " **************" << endl;

                //projection matrix of second camera: P1  = K[R|t]
                composeProjectionMat(T, R, P1);

                //triangulate from Richard Hartley and Andrew Zisserman
                TriangulatePointsHZ( P0, P1, normPoints2D_L, normPoints2D_R, 20, pcloud1);

                double reproj_error_L = calculateReprojectionErrorHZ(P0, normPoints2D_L, pcloud1);
                double reproj_error_R = calculateReprojectionErrorHZ(P1, normPoints2D_R, pcloud1);

                //check if pointa are triangulated --in front-- of both cameras. If yes break loop
                if (positionCheck(P0, pcloud1) && positionCheck(P1, pcloud1)) {
                    cout << "############## use this perspective Matrix " << "P"<< i << j << "################" << endl;
                    cout << "HZ: reprojection ERROR:  left:  " <<  reproj_error_L << "  right  " << reproj_error_R << endl;
                    foundPerspectiveMatrix = true;
                    break;
                }
                ++counter;
            }
        }

        if (4 == counter) {
            cout << "Shit. Can't found any right perspective Mat" << endl;
            return false;
        }

        for (unsigned int i=0; i<pcloud1.size(); i++) {
            outCloud.push_back(pcloud1[i]);
        }
    }

    return true;
}

bool positionCheck(const cv::Matx34f& P, const std::vector<cv::Point3f>& points3D) {
    vector<cv::Point3f> pcloud_pt3d_projected(points3D.size());

    cv::Matx44f P4x4 = cv::Matx44f::eye();
    for(int i=0;i<12;i++) P4x4.val[i] = P.val[i];

    cv::perspectiveTransform(points3D, pcloud_pt3d_projected, P4x4);

    // status of 3d points..infront=1 or behind=0
    vector<uchar> status;
    for (unsigned int i=0; i<points3D.size(); i++) {
        status.push_back((pcloud_pt3d_projected[i].z > 0) ? 1 : 0);
    }
    int count = cv::countNonZero(status);

    double percentage = ((double)count / (double)points3D.size());
    std::cout << count << "/" << points3D.size() << " = " << percentage*100.0 << "% are in front of camera" << std::endl;
    if(percentage < 0.55){
        //less than 55% of the points are in front of the camera
        return false;
    }
    return true;
}

bool DecomposeEtoRandT(const cv::Mat& E,
                       cv::Mat_<double>& R1,
                       cv::Mat_<double>& R2,
                       cv::Mat_<double>& t1,
                       cv::Mat_<double>& t2)
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

    cv::Mat_<double> R = svd_u * cv::Mat(w) * svd_vt; // HZ 9.19
    cv::Mat_<double> T = svd_u.col(2); // u3

    if (!CheckCoherentRotation(R)) {
        std::cout << "resulting rotation is not coherent" << std::endl;
        return 0;
    }

    // P' the second camera matrix, in the form of R|t
    // (rotation & translation)
    cv::Matx34d P1;
    P1 = cv::Matx34d(R(0, 0), R(0, 1), R(0, 2), T(0),
                     R(1, 0), R(1, 1), R(1, 2), T(1),
                     R(2, 0), R(2, 1), R(2, 2), T(2));
#endif

    //Using HZ E decomposition
    cv::SVD svd(E, cv::SVD::MODIFY_A);
    cv::Mat svd_u = svd.u;
    cv::Mat svd_vt = svd.vt;
    cv::Mat svd_w = svd.w;

    //check if first and second singular values are the same (as they should be)
    double singular_values_ratio = fabsf(svd_w.at<double>(0) / svd_w.at<double>(1));
    if(singular_values_ratio>1.0) singular_values_ratio = 1.0/singular_values_ratio; // flip ratio to keep it [0,1]
    if (singular_values_ratio < 0.7) {
        cout << "singular values are too far apart\n";
        return false;
    }

    //HZ 9.13
    cv::Matx33d W(0,-1,0,
                  1,0,0,
                  0,0,1);

    cv::Matx33d Wt(0,1,0,
                   -1,0,0,
                   0,0,1);

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
    if(fabsf(determinant(R))-1.0 > 1e-07) {
        cerr<<"det(R) != +-1.0, this is not a rotation matrix"<<endl;
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

    //get Inlier
    for(unsigned i = 0; i<points1.size(); ++i){
        if (inliers_fundamental[i] == 1) {
            cv::Mat point1 (homogenouse1[i]);
            cv::Mat point2 (homogenouse2[i]);
            cv::transpose(point2, point2);
            point1.convertTo(point1, CV_64F);
            point2.convertTo(point2, CV_64F);

            if ((point2 * F * point1).s[0] <= 0.1){
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
void loadIntrinsic(cv::Mat& K_L, cv::Mat& K_R, cv::Mat& distCoeff_L, cv::Mat& distCoeff_R) {
    //-----------------------------------------------------------------------------
    cv::FileStorage fs("data/calibration/final/intrinsic.yml", cv::FileStorage::READ);
    fs["cameraMatrixLeft"] >> K_L;
    fs["cameraMatrixRight"] >> K_R;
    fs["distCoeffsLeft"] >> distCoeff_L;
    fs["distCoeffsRight"] >> distCoeff_R;
    fs.release();
}

//-----------------------------------------------------------------------------
void loadExtrinsic(cv::Mat& R, cv::Mat& T, cv::Mat& E, cv::Mat& F ) {
    //-----------------------------------------------------------------------------
    cv::FileStorage fs("data/calibration/final/extrinsic.yml", cv::FileStorage::READ);
    fs["R"] >> R;
    fs["T"] >> T;
    fs["E"] >> E;
    fs["F"] >> F;
    fs.release();
}


void getScaleFactor(const cv::Mat& P0, const cv::Mat& P_LR, const cv::Mat& P_L, const cv::Mat& P_R, const vector<cv::Point2f>& normPoints_L1, const vector<cv::Point2f>&normPoints_R1, const vector<cv::Point2f>&normPoints_L2, const vector<cv::Point2f>& normPoints_R2, double& u, double& v) {
    std::vector<cv::Point3f> X, X_L, X_R;
    TriangulatePointsHZ(P0, P_LR, normPoints_L1, normPoints_R1, 5, X);
    TriangulatePointsHZ(P0, P_L , normPoints_L1, normPoints_L2, 5, X_L);
    TriangulatePointsHZ(P0, P_R , normPoints_R1, normPoints_R2, 5, X_R);

    double sum_L = 0;
    double sum_R = 0;
    for (unsigned int i = 0; i < X.size(); ++i) {
        sum_L += ((cv::norm(X[i])*1.0) / cv::norm(X_L[i])*1.0);
        sum_R += ((cv::norm(X[i])*1.0) / cv::norm(X_R[i])*1.0);
    }

    u = 1.0/X.size() * sum_L;
    v = 1.0/X.size() * sum_R;
}

void getScaleFactor2(const cv::Mat& T_LR, const cv::Mat& R_LR, const cv::Mat& T_L, const cv::Mat& R_L, const cv::Mat& T_R,  double& u, double& v) {
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
    u = x.at<double>(0,0);
    v = x.at<double>(1,0);
}

