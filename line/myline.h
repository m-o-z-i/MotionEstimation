#ifndef MYLINE_H
#define MYLINE_H

#include <opencv2/opencv.hpp>
class myLine; 


class myLine
{
public:
	myLine();
	myLine(CvPoint a, CvPoint b);
	
	double getLength();
	double getAngle();

private:
	CvPoint m_a;
	CvPoint m_b;
	double m_length;
	double m_angle;
};

#endif // MYLINE_H
