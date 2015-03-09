#include "FindCameraMatrices.h"
#include "Triangulation.h"


bool getRightProjectionMat(  cv::Mat& E,
                             const cv::Mat K,
                             const cv::Mat KInv,
                             const cv::Mat distCoeff,
                             cv::Mat& P1,
                             const vector<cv::Point2f>& points2D_1,
                             const vector<cv::Point2f>& points2D_2,
                             std::vector<cv::Point3f>& outCloud)
{
    // no rotation or translation for the left projection matrix
    //projection matrix of first camera P0 = K[I|0]
    cv::Mat P0 = (cv::Mat_<double>(3,4) <<
                  1.0, 0.0, 0.0, 0.0,
                  0.0, 1.0, 0.0, 0.0,
                  0.0, 0.0, 1.0, 0.0 );
    P0 = K * P0;

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
        //        if(determinant(R1)+1.0 < 1e-05) {
        //            //according to http://en.wikipedia.org/wiki/Essential_matrix#Showing_that_it_is_valid
        //            cout << "det(R) == -1 ["<<determinant(R1)<<"]: flip E's sign" << endl;
        //            E = -E;
        //            DecomposeEtoRandT(E,R1,R2,t1,t2);
        //        }
        if (!CheckCoherentRotation(R1) && !CheckCoherentRotation(R2)) {
            cout << "resulting rotations are not coherent\n";
            return false;
        }

        std::vector<cv::Mat_<double>> Rotations{R1,R2};
        std::vector<cv::Mat_<double>> Translations{t1,t2};

        int counter = 0;
        std::vector<cv::Point3f> pcloud, pcloud1, worldCoordinates;
        double reproj_error1, reproj_error2;
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
                P1 = (cv::Mat_<double>(3,4) <<
                      R(0,0),	R(0,1),	R(0,2),	T(0),
                      R(1,0),	R(1,1),	R(1,2),	T(1),
                      R(2,0),	R(2,1),	R(2,2),	T(2));



#if 0 //triangulations methods
                //triangulate Stereo
                triangulate(P0, P1, inliersF1, inliersF2, worldCoordinates);
                //calculate reprojection error
                vector<cv::Point3f> p0_r, p1_r;         //reprojected cartesian image coordinate with depth value
                vector<cv::Point2f> p0_err, p1_err;     //difference between original and reprojected 2D coordinates
                cv::Point2f avgReprojectionError;
                computeReprojectionError(P0, inliersF1, worldCoordinates, p0_r, p0_err, avgReprojectionError);
                cv::Point2f avgReprojectionError1;
                computeReprojectionError(P1, inliersF2, worldCoordinates, p1_r, p1_err, avgReprojectionError1);
                cout << "STEREO: reprojection ERROR:  left:  " << cv::norm(avgReprojectionError) << "  right  " << cv::norm(avgReprojectionError1) << endl;

                //triangulate OpenCV
                TriangulateOpenCV( P0, P1, inliersF1, inliersF2, pcloud);
                cv::Point2f avgReprojectionErrorOpenCV1;
                p0_r.clear(); p1_r.clear(); p0_err.clear(), p1_err.clear();
                computeReprojectionError(P0, inliersF1, pcloud, p0_r, p0_err, avgReprojectionErrorOpenCV1);
                cv::Point2f avgReprojectionErrorOpenCV2;
                computeReprojectionError(P1, inliersF2, pcloud, p1_r, p1_err, avgReprojectionErrorOpenCV2);
                double reproj_err_L = calculateReprojectionErrorOpenCV(P0, K, distCoeff,inliersF1, pcloud);
                double reproj_err_R = calculateReprojectionErrorOpenCV(P1, K, distCoeff,inliersF2, pcloud);
                cout << "OPENCV: reprojection ERROR:  left:  " <<  cv::norm(avgReprojectionErrorOpenCV1) << " or " << reproj_err_L << "  right  " << cv::norm(avgReprojectionErrorOpenCV2) << " or " << reproj_err_R << endl;
#endif

                //triangulate Richard Hartley and Andrew Zisserman
                TriangulatePointsHZ( P0, P1, points2D_1, points2D_2, KInv, pcloud1);
//                cv::Point2f avgReprojectionErrorHZ1;
//                p0_r.clear(); p1_r.clear(); p0_err.clear(), p1_err.clear();
//                computeReprojectionError(P0, inliersF1, pcloud1, p0_r, p0_err, avgReprojectionErrorHZ1);
//                cv::Point2f avgReprojectionErrorHZ2;
//                computeReprojectionError(P1, inliersF2, pcloud1, p1_r, p1_err, avgReprojectionErrorHZ2);
                double reproj_error_L = calculateReprojectionErrorHZ(P0, K, points2D_1, pcloud1);
                double reproj_error_R = calculateReprojectionErrorHZ(P1, K, points2D_2, pcloud1);
                //cout << "HZ: reprojection ERROR:  left:  " <<  reproj_error_L << "  right  " << reproj_error_R << endl;


                //determine if points are in front of both cameras
//                uint pointsInFront = 0;
//                for(uint i = 0; i < inliersF1.size(); i++) {
//                    //cout << "Point " << i << ":\n";
//                    //cout << "Original 2D coordinate: p: " << p0[i].x << ", " << p0[i].y << "  and  p': " << p1[i].x << ", " << p1[i].y << endl;
//                    //cout << "3D coordinate:  X: " << worldCoordinates[i].x << ", " << worldCoordinates[i].y << ", " << worldCoordinates[i].z <<  endl;
//                    //cout << "Reprojected 2D coordinates: x: " << p0_r[i].x << ", " << p0_r[i].y << "  and  x': " << p1_r[i].x << ", " << p1_r[i].y << "  depth0: " << p0_r[i].z << ", depth1: " << p1_r[i].z << endl;
//                    //cout << "Reprojection error: " << p0_err[i].x << ", " << p0_err[i].y <<  "  and  x': " << p1_err[i].x << ", " << p1_err[i].y << endl;

//                    double depth0 = p0_r[i].z; //depth information from homogeneous coordinate w (z-buffer style)
//                    double depth1 = p1_r[i].z;
//                    if (depth0 > 0 && depth1 > 0) { //valid camera configuration
//                        pointsInFront++;;
//                    }
//                }

                //check if pointa are triangulated --in front-- of both cameras. If yes break loop
                if (TestTriangulation(P0, pcloud1) && TestTriangulation(P1, pcloud1)) {
                    cout << "############## use this perspective Matrix ################" << endl;
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

bool TestTriangulation(const cv::Matx34f& P, const std::vector<cv::Point3f>& points3D) {
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
    //std::cout << count << "/" << points3D.size() << " = " << percentage*100.0 << "% are in front of camera" << std::endl;
    if(percentage < 0.55){
        //less than 75% of the points are in front of the camera
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

void getFundamentalMatrix(pair<vector<cv::Point2f>, vector<cv::Point2f>> const& points, vector<cv::Point2f> *inliers1, vector<cv::Point2f> *inliers2, cv::Mat& F) {
    // Compute F matrix using RANSAC
    if(points.first.size() != points.second.size()){
        return;
    }

    //vector<cv::Point2f> p1;
    //vector<cv::Point2f> p2;

    std::vector<uchar> inliers_fundamental(points.first.size(),0);
    F = cv::findFundamentalMat(
            cv::Mat(points.first), cv::Mat(points.second),   // matching points
            inliers_fundamental,                             // match status (inlier ou outlier)
            cv::FM_RANSAC,                                   // RANSAC method
            0.1,                                             // distance to epipolar line
            0.99);                                           // confidence probability

    //get Inlier
    for(unsigned i = 0; i<points.first.size(); ++i){
        if (inliers_fundamental[i] == 1) {
            inliers1->push_back(points.first[i]);
            inliers2->push_back(points.second[i]);
        } else {
            inliers1->push_back(cv::Point2f(0,0));
            inliers2->push_back(cv::Point2f(0,0));
        }
    }
    // check x' * F * x = 0 ??
    //    vector<cv::Point3f> homogenouse1;
    //    vector<cv::Point3f> homogenouse2;
    //    cv::convertPointsToHomogeneous(p1, homogenouse1);
    //    cv::convertPointsToHomogeneous(p2, homogenouse2);

    //    for(unsigned i = 0; i<homogenouse1.size(); ++i){

    //        std::cout <<  cv::norm((homogenouse2[i].dot(F)).dot(homogenouse1[i])) << std::endl;
    //    }

}
