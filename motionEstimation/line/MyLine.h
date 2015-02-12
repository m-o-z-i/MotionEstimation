#ifndef MYLINE_H
#define MYLINE_H

#include <opencv2/opencv.hpp>

class MyLine; 

class MyLine
{
public:
	MyLine();
	MyLine(cv::Point2f a, cv::Point2f b);
	
	double getLength();
	double getAngle();
	cv::Point2f getPointA() const;
	cv::Point2f getPointB() const;

private:
	cv::Point2f m_a;
	cv::Point2f m_b;
	double m_length;
	double m_angle;
};

#endif // MYLINE_H
