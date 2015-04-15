#include "FindPoints.h"

inline static float square(int a)
{
    return a * a;
}

vector<cv::Point2f> getStrongFeaturePoints(const cv::Mat& image, int number, float minQualityLevel, float minDistance) {
    /* Shi and Tomasi Feature Tracking! */

    /* Preparation: This array will contain the features found in image1. */
    vector<cv::Point2f> image_features;

    /* Preparation: BEFORE the function call this variable is the array size
     * (or the maximum number of features to find).  AFTER the function call
     * this variable is the number of features actually found.
     */
    int number_of_features = number;

    /* Actually run the Shi and Tomasi algorithm!!
     * "Image" is the input image.
     * qualityLevel specifies the minimum quality of the features (based on the eigenvalues)
     * minDistance specifies the minimum Euclidean distance between features
     * RETURNS:
     * "image_features" will contain the feature points.
     */
    cv::goodFeaturesToTrack(image, image_features, number_of_features, minQualityLevel, minDistance);
    return image_features;
}

void refindFeaturePoints(cv::Mat const& prev_image, cv::Mat const& next_image, vector<cv::Point2f> frame1_features, vector<cv::Point2f> &points1, vector<cv::Point2f> &points2){
    /* Pyramidal Lucas Kanade Optical Flow! */

    /* This array will contain the locations of the points from frame 1 in frame 2. */
    vector<cv::Point2f>  frame2_features;

    /* The i-th element of this array will be non-zero if and only if the i-th feature of
     * frame 1 was found in frame 2.
     */
    vector<unsigned char> optical_flow_found_feature;

    /* The i-th element of this array is the error in the optical flow for the i-th feature
     * of frame1 as found in frame 2.  If the i-th feature was not found (see the array above)
     * I think the i-th entry in this array is undefined.
     */
    vector<float> optical_flow_feature_error;

    /* This is the window size to use to avoid the aperture problem (see slide "Optical Flow: Overview"). */
    CvSize optical_flow_window = cvSize(15,15);

    /* 0-based maximal pyramid level number; if set to 0, pyramids are not used (single level),
     * if set to 1, two levels are used, and so on; if pyramids are passed to input then algorithm
     * will use as many levels as pyramids have but no more than maxLevel.
     * */
    int maxLevel = 10;

    /* This termination criteria tells the algorithm to stop when it has either done 20 iterations or when
     * epsilon is better than .3.  You can play with these parameters for speed vs. accuracy but these values
     * work pretty well in many situations.
     */
    cv::TermCriteria optical_flow_termination_criteria
            = cv::TermCriteria( cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, .3 );

    /* Actually run Pyramidal Lucas Kanade Optical Flow!!
     * "prev_image" is the first frame with the known features. pyramid constructed by buildOpticalFlowPyramid()
     * "next_image" is the second frame where we want to find the first frame's features.
     * "frame1_features" are the features from the first frame.
     * "frame2_features" is the (outputted) locations of those features in the second frame.
     * "number_of_features" is the number of features in the frame1_features array.
     * "optical_flow_window" is the size of the window to use to avoid the aperture problem.
     * "maxLevel" is the maximum number of pyramids to use.  0 would be just one level.
     * "optical_flow_found_feature" is as described above (non-zero iff feature found by the flow).
     * "optical_flow_feature_error" is as described above (error in the flow for this feature).
     * "optical_flow_termination_criteria" is as described above (how long the algorithm should look).
     * "0" means disable enhancements.  (For example, the second array isn't pre-initialized with guesses.)
     */
    //TODO: improve TermCriteria. do not quit program when it is reached
    cv::calcOpticalFlowPyrLK(prev_image, next_image, frame1_features, frame2_features, optical_flow_found_feature,
                             optical_flow_feature_error, optical_flow_window, maxLevel,
                             optical_flow_termination_criteria, cv::OPTFLOW_LK_GET_MIN_EIGENVALS);


    vector<cv::Point2f>::iterator iter_f1 = frame1_features.begin();
    vector<cv::Point2f>::iterator iter_f2 = frame2_features.begin();
    for (unsigned i = 0; i < frame1_features.size(); ++i){
        if ( optical_flow_found_feature[i] == 0 ){
            frame1_features[i] = cv::Point2f(0,0);
            frame2_features[i] = cv::Point2f(0,0);
        }
        ++iter_f1;
        ++iter_f2;

        points1.push_back(frame1_features[i]);
        points2.push_back(frame2_features[i]);
    }
}

void getInliersFromMedianValue (const pair<vector<cv::Point2f>, vector<cv::Point2f> >& features, vector<cv::Point2f> &inliers1, vector<cv::Point2f> &inliers2){
    vector<float> directions;
    vector<float> lengths;

    for (unsigned int i = 0; i < features.first.size(); ++i){
        float direction = atan2( (float) (features.first[i].y - features.second[i].y) , (float) (features.first[i].x - features.second[i].x) );
        directions.push_back(direction);

        float length = sqrt( square(features.first[i].y - features.second[i].y) + square(features.first[i].x - features.second[i].x) );
        lengths.push_back(length);
    }

    sort(directions.begin(), directions.end());
    float median_direction = directions[(int)(directions.size()/2)];

    sort(lengths.begin(),lengths.end());
    float median_lenght = lengths[(int)(lengths.size()/2)];


    for(unsigned int j = 0; j < features.first.size(); ++j)
    {
        float direction = atan2( (float) (features.first[j].y - features.second[j].y) , (float) (features.first[j].x - features.second[j].x) );
        float length = sqrt( square(features.first[j].y - features.second[j].y) + square(features.first[j].x - features.second[j].x) );
        if (direction < median_direction + 0.05 && direction > median_direction - 0.05 && length < (median_lenght * 2) && length > (median_lenght * 0.5) ) {
            inliers1.push_back(features.first[j]);
            inliers2.push_back(features.second[j]);
        } else {
            inliers1.push_back(cv::Point2f(0,0));
            inliers2.push_back(cv::Point2f(0,0));
        }
    }
}

void getInliersFromHorizontalDirection (const pair<vector<cv::Point2f>, vector<cv::Point2f> >& features, vector<cv::Point2f> &inliers1, vector<cv::Point2f> &inliers2){
    for(unsigned int j = 0; j < features.first.size(); ++j)
    {
        float direction = atan2( (float) (features.first[j].y - features.second[j].y) , (float) (features.first[j].x - features.second[j].x) );

        if (fabs(direction) < 0.4) {
            inliers1.push_back(features.first[j]);
            inliers2.push_back(features.second[j]);
        } else {
            inliers1.push_back(cv::Point2f(0,0));
            inliers2.push_back(cv::Point2f(0,0));
        }
    }
}


void deleteUnvisiblePoints(vector<cv::Point2f>& points1L, vector<cv::Point2f>& points1La, vector<cv::Point2f>& points1R, vector<cv::Point2f>& points1Ra, vector<cv::Point2f>& points2L, vector<cv::Point2f>& points2R, int resX, int resY){


    int size = points1L.size();
    // iterate over all points and delete points, that are not in all frames visible;
    vector<cv::Point2f>::iterator iter_c1a = points1L.begin();
    vector<cv::Point2f>::iterator iter_c1b = points1R.begin();
    vector<cv::Point2f>::iterator iter_c2a = points1La.begin();
    vector<cv::Point2f>::iterator iter_c2b = points1Ra.begin();
    vector<cv::Point2f>::iterator iter_c3a = points2L.begin();
    vector<cv::Point2f>::iterator iter_c3b = points2R.begin();
    for (unsigned int i = 0; i < size ; ++i ) {
        if (1 >= points1L[iter_c1a-points1L.begin()].x   &&
                1 >= points1L[iter_c1a-points1L.begin()].y   ||
                1 >= points1La[iter_c2a-points1La.begin()].x &&
                1 >= points1La[iter_c2a-points1La.begin()].y ||
                1 >= points2L[iter_c3a-points2L.begin()].x &&
                1 >= points2L[iter_c3a-points2L.begin()].y ||
                1 >= points1R[iter_c1b-points1R.begin()].x   &&
                1 >= points1R[iter_c1b-points1R.begin()].y   ||
                1 >= points1Ra[iter_c2b-points1Ra.begin()].x &&
                1 >= points1Ra[iter_c2b-points1Ra.begin()].y ||
                1 >= points2R[iter_c3b-points2R.begin()].x &&
                1 >= points2R[iter_c3b-points2R.begin()].y ||

                resX <= points1L[iter_c1a-points1L.begin()].x   &&
                resY <= points1L[iter_c1a-points1L.begin()].y   ||
                resX <= points1La[iter_c2a-points1La.begin()].x &&
                resY <= points1La[iter_c2a-points1La.begin()].y ||
                resX <= points2L[iter_c3a-points2L.begin()].x &&
                resY <= points2L[iter_c3a-points2L.begin()].y ||
                resX <= points1R[iter_c1b-points1R.begin()].x   &&
                resY <= points1R[iter_c1b-points1R.begin()].y   ||
                resX <= points1Ra[iter_c2b-points1Ra.begin()].x &&
                resY <= points1Ra[iter_c2b-points1Ra.begin()].y ||
                resX <= points2R[iter_c3b-points2R.begin()].x &&
                resY <= points2R[iter_c3b-points2R.begin()].y )
        {
            points1L.erase(iter_c1a);
            points1R.erase(iter_c1b);
            points1La.erase(iter_c2a);
            points1Ra.erase(iter_c2b);
            points2L.erase(iter_c3a);
            points2R.erase(iter_c3b);
        } else
        {
            ++iter_c1a;
            ++iter_c1b;
            ++iter_c2a;
            ++iter_c2b;
            ++iter_c3a;
            ++iter_c3b;
        }
    }
}

void deleteZeroLines(vector<cv::Point2f>& points1, vector<cv::Point2f>& points2){
    int size = points1.size();
    vector<cv::Point2f>::iterator iter_p1 = points1.begin();
    vector<cv::Point2f>::iterator iter_p2 = points2.begin();
    for (unsigned int i = 0; i < size; ++i) {
        if ((0 == points1[iter_p1-points1.begin()].x && 0 == points1[iter_p1-points1.begin()].y) ||
                (0 == points2[iter_p2-points2.begin()].x && 0 == points2[iter_p2-points2.begin()].y)){
            points1.erase(iter_p1);
            points2.erase(iter_p2);
        } else {
            ++iter_p1;
            ++iter_p2;
        }
    }
}

void deleteZeroLines(vector<cv::Point2f>& points1La, vector<cv::Point2f>& points1Lb, vector<cv::Point2f>& points1Ra,
                     vector<cv::Point2f>& points1Rb, vector<cv::Point2f>& points2L, vector<cv::Point2f>& points2R){
    int size = points1La.size();
    vector<cv::Point2f>::iterator iter_p1La = points1La.begin();
    vector<cv::Point2f>::iterator iter_p1Lb = points1Lb.begin();
    vector<cv::Point2f>::iterator iter_p1Ra = points1Ra.begin();
    vector<cv::Point2f>::iterator iter_p1Rb = points1Rb.begin();
    vector<cv::Point2f>::iterator iter_p2L  = points2L.begin();
    vector<cv::Point2f>::iterator iter_p2R  = points2R.begin();
    for (unsigned int i = 0; i < size; ++i) {
        if ((0 == points1La[iter_p1La-points1La.begin()].x && 0 == points1La[iter_p1La-points1La.begin()].y) ||
                (0 == points1Lb[iter_p1Lb-points1Lb.begin()].x && 0 == points1Lb[iter_p1Lb-points1Lb.begin()].y) ||
                (0 == points1Ra[iter_p1Ra-points1Ra.begin()].x && 0 == points1Ra[iter_p1Ra-points1Ra.begin()].y) ||
                (0 == points1Rb[iter_p1Rb-points1Rb.begin()].x && 0 == points1Rb[iter_p1Rb-points1Rb.begin()].y) ||
                (0 == points2L[iter_p2L-points2L.begin()].x && 0 == points2L[iter_p2L-points2L.begin()].y) ||
                (0 == points2R[iter_p2R-points2R.begin()].x && 0 == points2R[iter_p2R-points2R.begin()].y))
        {
            points1La.erase(iter_p1La);
            points1Lb.erase(iter_p1Lb);
            points1Ra.erase(iter_p1Ra);
            points1Rb.erase(iter_p1Rb);
            points2L.erase(iter_p2L);
            points2R.erase(iter_p2R);
        } else {
            ++iter_p1La ;
            ++iter_p1Lb ;
            ++iter_p1Ra ;
            ++iter_p1Rb ;
            ++iter_p2L  ;
            ++iter_p2R  ;
        }
    }
}

void deleteZeroLines(vector<cv::Point2f>& points1L, vector<cv::Point2f>& points1R,
                     vector<cv::Point2f>& points2L, vector<cv::Point2f>& points2R){
    int size = points1L.size();
    vector<cv::Point2f>::iterator iter_p1L = points1L.begin();
    vector<cv::Point2f>::iterator iter_p1R = points1R.begin();
    vector<cv::Point2f>::iterator iter_p2L  = points2L.begin();
    vector<cv::Point2f>::iterator iter_p2R  = points2R.begin();
    for (unsigned int i = 0; i < size; ++i) {
        if ((0 == points1L[iter_p1L-points1L.begin()].x && 0 == points1L[iter_p1L-points1L.begin()].y) ||
                (0 == points1R[iter_p1R-points1R.begin()].x && 0 == points1R[iter_p1R-points1R.begin()].y) ||
                (0 == points2L[iter_p2L-points2L.begin()].x && 0 == points2L[iter_p2L-points2L.begin()].y) ||
                (0 == points2R[iter_p2R-points2R.begin()].x && 0 == points2R[iter_p2R-points2R.begin()].y))
        {
            points1L.erase(iter_p1L);
            points1R.erase(iter_p1R);
            points2L.erase(iter_p2L);
            points2R.erase(iter_p2R);
        } else {
            ++iter_p1L ;
            ++iter_p1R ;
            ++iter_p2L  ;
            ++iter_p2R  ;
        }
    }
}

void deleteZeroLines(vector<cv::Point2f>& points1L, vector<cv::Point2f>& points1R,
                     vector<cv::Point2f>& points2L, vector<cv::Point2f>& points2R,
                     vector<cv::Point3f>& cloud1, vector<cv::Point3f>& cloud2 )
{
    int size = points1L.size();
    vector<cv::Point2f>::iterator iter_p1L = points1L.begin();
    vector<cv::Point2f>::iterator iter_p1R = points1R.begin();
    vector<cv::Point2f>::iterator iter_p2L  = points2L.begin();
    vector<cv::Point2f>::iterator iter_p2R  = points2R.begin();
    vector<cv::Point3f>::iterator iter_cloud1  = cloud1.begin();
    vector<cv::Point3f>::iterator iter_cloud2  = cloud2.begin();
    for (unsigned int i = 0; i < size; ++i) {
        if ((0 == points1L[iter_p1L-points1L.begin()].x && 0 == points1L[iter_p1L-points1L.begin()].y) ||
                (0 == points1R[iter_p1R-points1R.begin()].x && 0 == points1R[iter_p1R-points1R.begin()].y) ||
                (0 == points2L[iter_p2L-points2L.begin()].x && 0 == points2L[iter_p2L-points2L.begin()].y) ||
                (0 == points2R[iter_p2R-points2R.begin()].x && 0 == points2R[iter_p2R-points2R.begin()].y) ||
                (0 == cloud1[iter_cloud1-cloud1.begin()].x  && 0 == cloud1[iter_cloud1-cloud1.begin()].y)  ||
                (0 == cloud2[iter_cloud2-cloud2.begin()].x  && 0 == cloud2[iter_cloud2-cloud2.begin()].y))
        {
            points1L.erase(iter_p1L);
            points1R.erase(iter_p1R);
            points2L.erase(iter_p2L);
            points2R.erase(iter_p2R);
            cloud1.erase(iter_cloud1);
            cloud2.erase(iter_cloud2);
        } else {
            ++iter_p1L ;
            ++iter_p1R ;
            ++iter_p2L  ;
            ++iter_p2R  ;
            ++iter_cloud1  ;
            ++iter_cloud2  ;
        }
    }
}


void normalizePoints(const cv::Mat& KInv, const vector<cv::Point2f>& points1, const vector<cv::Point2f>& points2, vector<cv::Point2f>& normPoints1, vector<cv::Point2f>& normPoints2){

    vector<cv::Point3f> points1_h, points2_h;
    cv::convertPointsToHomogeneous(points1, points1_h);
    cv::convertPointsToHomogeneous(points2, points2_h);

    KInv.convertTo(KInv, CV_32F);

    for(unsigned int i = 0; i < points1.size(); ++i){
        cv::Mat matPoint1_h(points1_h[i]);
        matPoint1_h.convertTo(matPoint1_h, CV_32F);

        cv::Mat matPoint2_h(points2_h[i]);
        matPoint2_h.convertTo(matPoint2_h, CV_32F);

        points1_h[i] = cv::Point3f(cv::Mat(KInv * matPoint1_h));
        points2_h[i] = cv::Point3f(cv::Mat(KInv * matPoint2_h));
    }
    cv::convertPointsFromHomogeneous(points1_h, normPoints1);
    cv::convertPointsFromHomogeneous(points2_h, normPoints2);
}

void normalizePoints(const cv::Mat& KLInv, const cv::Mat& KRInv, const vector<cv::Point2f>& points_L, const vector<cv::Point2f>& points_R, vector<cv::Point2f>& normPoints_L, vector<cv::Point2f>& normPoints_R){

    vector<cv::Point3f> points_Lh, points_Rh;
    cv::convertPointsToHomogeneous(points_L, points_Lh);
    cv::convertPointsToHomogeneous(points_R, points_Rh);

    for(unsigned int i = 0; i < points_L.size(); ++i){
        cv::Mat matPoint_Lh(points_Lh[i]);
        cv::Mat matPoint_Rh(points_Rh[i]);

        points_Lh[i] = cv::Point3f(cv::Mat(KLInv * matPoint_Lh));
        points_Rh[i] = cv::Point3f(cv::Mat(KRInv * matPoint_Rh));
    }
    cv::convertPointsFromHomogeneous(points_Lh, normPoints_L);
    cv::convertPointsFromHomogeneous(points_Rh, normPoints_R);
}


void findCorresPoints_LucasKanade(const cv::Mat& frame_L1, const cv::Mat& frame_R1, const cv::Mat& frame_L2, const cv::Mat& frame_R2, vector<cv::Point2f> &points_L1, vector<cv::Point2f> &points_R1, vector<cv::Point2f> &points_L2, vector<cv::Point2f> &points_R2){
    // find corresponding points
    vector<cv::Point2f> points_L1_temp, points_R1_temp, points_L1a_temp, points_R1a_temp, points_L2_temp, points_R2_temp;
    vector<cv::Point2f> features = getStrongFeaturePoints(frame_L1, 20,0.001,5);

    if (0 == features.size()){
        return;
    }

    refindFeaturePoints(frame_L1, frame_R1, features, points_L1_temp, points_R1_temp);
    refindFeaturePoints(frame_L1, frame_L2, points_L1_temp, points_L1a_temp, points_L2_temp);
    refindFeaturePoints(frame_R1, frame_R2, points_R1_temp, points_R1a_temp, points_R2_temp);

    //    drawPoints(frame_L1, features, "feaures left found" , cv::Scalar(2,55,212));
    //    drawPoints(frame_R1, points_R1_temp, "feaures right found" , cv::Scalar(2,55,212));


    // delete in all frames points, that are not visible in each frames
    deleteUnvisiblePoints(points_L1_temp, points_L1a_temp, points_R1_temp, points_R1a_temp, points_L2_temp, points_R2_temp, frame_L1.cols, frame_L1.rows);


    for (unsigned int i = 0; i < points_L1_temp.size(); ++i){
        points_L1.push_back(points_L1_temp[i]);
        points_R1.push_back(points_R1_temp[i]);
        points_L2.push_back(points_L2_temp[i]);
        points_R2.push_back(points_R2_temp[i]);
    }
}

void fastFeatureMatcher(const cv::Mat& frame_L1, const cv::Mat& frame_R1, const cv::Mat& frame_L2, const cv::Mat& frame_R2, vector<cv::Point2f> &points_L1, vector<cv::Point2f>& points_R1, vector<cv::Point2f> &points_L2, vector<cv::Point2f> &points_R2) {
    vector<cv::DMatch> matches;

    vector<cv::KeyPoint>left_keypoints,right_keypoints;

    // Detect keypoints in the left and right images
    cv::FastFeatureDetector ffd;
    ffd.detect(frame_L1, left_keypoints);
    ffd.detect(frame_R1, right_keypoints);

    vector<cv::Point2f>left_points;
    KeyPointsToPoints(left_keypoints,left_points);

    vector<cv::Point2f>right_points(left_points.size());

    // Calculate the optical flow field:
    //  how each left_point moved across the 2 images
    vector<uchar>vstatus; vector<float>verror;
    cv::calcOpticalFlowPyrLK(frame_L1, frame_R1, left_points, right_points, vstatus, verror);

    // First, filter out the points with high error
    vector<cv::Point2f>right_points_to_find;
    vector<int>right_points_to_find_back_index;
    for (unsigned int i=0; i<vstatus.size(); i++) {
        if (vstatus[i] &&verror[i] < 12.0) {
            // Keep the original index of the point in the
            // optical flow array, for future use
            right_points_to_find_back_index.push_back(i);
            // Keep the feature point itself
            right_points_to_find.push_back(right_points[i]);
        } else {
            vstatus[i] = 0; // a bad flow
        }
    }

    drawCorresPoints(frame_L1, left_points, right_points, "left right fast", cv::Scalar(255,0,0));

    // for each right_point see which detected feature it belongs to
    cv::Mat right_points_to_find_flat = cv::Mat(right_points_to_find).reshape(1,right_points_to_find.size()); //flatten array

    vector<cv::Point2f>right_features; // detected features
    KeyPointsToPoints(right_keypoints,right_features);

    cv::Mat right_features_flat = cv::Mat(right_features).reshape(1,right_features.size());

    //FlannBasedMatcher matcher;

    // Look around each OF point in the right image
    //  for any features that were detected in its area
    //  and make a match.
    cv::BFMatcher matcher(CV_L2);
    vector<vector<cv::DMatch>>nearest_neighbors;
    matcher.radiusMatch(
                right_points_to_find_flat,
                right_features_flat,
                nearest_neighbors,
                2.0f);

    // Check that the found neighbors are unique (throw away neighbors
    //  that are too close together, as they may be confusing)
    std::set<int>found_in_right_points; // for duplicate prevention
    for(int i=0;i<nearest_neighbors.size();i++) {
        cv::DMatch _m;
        if(nearest_neighbors[i].size()==1) {
            _m = nearest_neighbors[i][0]; // only one neighbor
        } else if(nearest_neighbors[i].size()>1) {
            // 2 neighbors – check how close they are
            float ratio = nearest_neighbors[i][0].distance / nearest_neighbors[i][1].distance;
            if(ratio < 0.7) { // not too close
                // take the closest (first) one
                _m = nearest_neighbors[i][0];
            } else { // too close – we cannot tell which is better
                continue; // did not pass ratio test – throw away
            }
        } else {
            continue; // no neighbors... :(
        }

        // prevent duplicates
        if (found_in_right_points.find(_m.trainIdx) == found_in_right_points.end()) {
            // The found neighbor was not yet used:
            // We should match it with the original indexing
            // ofthe left point
            _m.queryIdx = right_points_to_find_back_index[_m.queryIdx];
            matches.push_back(_m); // add this match
            found_in_right_points.insert(_m.trainIdx);
        }
    }
    cout<<"pruned "<< matches.size() <<" / "<<nearest_neighbors.size() <<" matches"<<endl;

    cv::Mat img_out;
    cv::drawMatches(frame_L1, left_keypoints, frame_R1, right_keypoints, matches, img_out);
    cv::imshow("test fast matches", img_out);
    cv::waitKey();
}
