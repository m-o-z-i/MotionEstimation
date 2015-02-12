#include "MyLine.h"

inline static double square(int a)
{
	return a * a;
}

MyLine::MyLine()
{
}

MyLine::MyLine(cv::Point2f a, cv::Point2f b)
	:m_a(a),m_b(b)
{
	m_angle = atan2( (double) m_a.y - m_b.y, (double) m_b.x - m_b.x );
	m_length = sqrt( square(m_a.y - m_b.y) + square(m_a.x - m_b.x) );
}

double MyLine::getLength(){
	return m_length;
}
double MyLine::getAngle(){
	return m_angle;
}

cv::Point2f MyLine::getPointA() const{
	return m_a;
}

cv::Point2f MyLine::getPointB() const{
	return m_b;
}

