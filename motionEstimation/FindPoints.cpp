#include "FindPoints.h"

inline static double square(int a)
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

pair<vector<cv::Point2f>, vector<cv::Point2f>> refindFeaturePoints(const cv::Mat& prev_image, const cv::Mat& next_image, vector<cv::Point2f> frame1_features){
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
            frame1_features.erase(iter_f1);
            frame2_features.erase(iter_f2);
        }
        ++iter_f1;
        ++iter_f2;
    }


    return make_pair(frame1_features, frame2_features);
}

void getInliersFromMeanValue (const pair<vector<cv::Point2f>, vector<cv::Point2f> >& features, vector<cv::Point2f>* inliers1, vector<cv::Point2f>* inliers2){
    vector<double> directions;
    vector<double> lengths;

    for (unsigned i = 0; i < features.first.size(); ++i){
        double direction = atan2( (double) features.first[i].y - features.second[i].y, (double) features.second[i].x - features.second[i].x );
        directions.push_back(direction);

        double length = sqrt( square(features.first[i].y - features.second[i].y) + square(features.first[i].x - features.second[i].x) );
        lengths.push_back(length);
    }

    sort(directions.begin(), directions.end());
    double median_angle = directions[(int)(directions.size()/2)];

    sort(lengths.begin(),lengths.end());
    double median_lenght = lengths[(int)(lengths.size()/2)];

    for(unsigned i = 0; i < features.first.size(); ++i)
    {
        double direction = atan2( (double) features.first[i].y - features.second[i].y, (double) features.second[i].x - features.second[i].x );
        double length = sqrt( square(features.first[i].y - features.second[i].y) + square(features.first[i].x - features.second[i].x) );

        if (direction < median_angle + 2 && direction > median_angle - 2 ) {
            if (length < (median_lenght*3) && length > 1.5 && length > median_lenght*0.1) {
                inliers1->push_back(features.first[i]);
                inliers2->push_back(features.second[i]);
            }
        }
    }
}
