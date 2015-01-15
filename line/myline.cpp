#include "myline.h"

inline static double square(int a)
{
	return a * a;
}

myLine::myLine()
{
}

myLine::myLine(CvPoint a, CvPoint b)
	:m_a(a),m_b(b)
{
	m_angle = atan2( (double) m_a.y - m_b.y, (double) m_b.x - m_b.x );
	m_length = sqrt( square(m_a.y - m_b.y) + square(m_a.x - m_b.x) );
}

double myLine::getLength(){
	return m_length;
}
double myLine::getAngle(){
	return m_angle;
}
