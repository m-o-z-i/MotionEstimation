#include "FindCameraMatrices.h"
#include "Triangulation.h"


bool getRightProjectionMat(  cv::Mat& E,
                             const cv::Mat K,
                             const cv::Mat KInv,
                             const cv::Mat distCoeff,
                             const vector<cv::Point2f>& inliersF1,
                             const vector<cv::Point2f>& inliersF2,
                             cv::Matx34f& P1,
                             std::vector<cv::Point3f>& outCloud)
{
    // no rotation or translation for the left projection matrix
    cv::Matx34f P0(1,0,0,0,
                   0,1,0,0,
                   0,0,1,0);

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
        std::vector<cv::Point3f> pcloud, pcloud1;
        double reproj_error1, reproj_error2;

        // find right solution of 4 possible translations and rotations
        for (unsigned int i = 0; i < 2; ++i) {
            cv::Mat_<double> R = Rotations[i];
            if (!CheckCoherentRotation(R)) {
                cout << "resulting rotation R is not coherent\n";
                counter += 2;
                continue;
            }

            for (unsigned int j = 0; j < 2; ++j) {
                pcloud.clear(); pcloud1.clear();
                cv::Mat_<double> T = Translations[j];

                reproj_error1 = TriangulateOpenCV(K, distCoeff, inliersF1, inliersF2, P0, P1, pcloud);

                reproj_error2 = TriangulatePoints(inliersF1, inliersF2,K, KInv,P0,P1,pcloud);

                cout << "projection ERROR: CV:  " << reproj_error1 << "  other: " << reproj_error2 << std::endl;

                //check if pointa are triangulated --in front-- of both cameras. If yes break loop
                if (TestTriangulation(pcloud,P1) && reproj_error1 < 100.0) {
                    break;
                }
                ++counter;
            }
        }

        if (4 == counter) {
            cout << "Shit." << endl;
            return false;
        }

        for (unsigned int i=0; i<pcloud.size(); i++) {
            outCloud.push_back(pcloud[i]);
        }
    }

    return true;
}

bool TestTriangulation(const std::vector<cv::Point3f>& pcloud_pt3d, const cv::Matx34f& P) {
    vector<cv::Point3f> pcloud_pt3d_projected(pcloud_pt3d.size());

    cv::Matx44f P4x4 = cv::Matx44f::eye();
    for(int i=0;i<12;i++) P4x4.val[i] = P.val[i];

    cv::perspectiveTransform(pcloud_pt3d, pcloud_pt3d_projected, P4x4);

    // status of 3d points..infront=1 or behind=0
    vector<uchar> status;
    for (unsigned int i=0; i<pcloud_pt3d.size(); i++) {
        status.push_back((pcloud_pt3d_projected[i].z > 0) ? 1 : 0);
    }
    int count = cv::countNonZero(status);

    double percentage = ((double)count / (double)pcloud_pt3d.size());
    std::cout << count << "/" << pcloud_pt3d.size() << " = " << percentage*100.0 << "% are in front of camera" << std::endl;
    if(percentage < 0.75)
        return false; //less than 75% of the points are in front of the camera

    return true;
}

bool DecomposeEtoRandT(cv::Mat E,
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
            //p1.push_back(points.first[i]);
            //p2.push_back(points.second[i]);
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
