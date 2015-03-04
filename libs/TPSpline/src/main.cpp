#include<iostream>
#include<fstream>
#include <cv.h>
#include <cxcore.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdlib.h> 
#include "CThinPlateSpline.h"

std::vector<cv::Point> readPList(const char*);
cv::Mat readImage(const char*);
void writeImage(cv::Mat,const char*);

int main(int argc, char **argv)
{
	if( argc < 4)
    {
		std::cout <<" Not enought parameters, please specify pS, pD, src" << std::endl;
		return -1;
    }
	std::cout<<"hey dear, im here.."<<std::endl;
	std::vector<cv::Point> pS;
	std::vector<cv::Point> pD;

	// load source and destination lists
	pS = readPList(argv[1]);
	pD = readPList(argv[2]);
	std::cout<<"hey dear, im here.."<<std::endl;
	// load images
	cv::Mat img = readImage(argv[3]);

	std::cout<<"hey dear, im here.."<<std::endl;
	// Thin plate spline
	CThinPlateSpline tps(pS,pD);
	Mat dst;
	tps.warpImage(img,dst,0.01,INTER_CUBIC,BACK_WARP);


	//output images
	if( argc >= 4)
		writeImage(dst,argv[4]);

	return 0;
}

std::vector<cv::Point> readPList(const char* fileName)
{
	std::ifstream infile(fileName);
	if (! infile) 
	{
		std::cerr << "unable to open the file  for reading" << std::endl;
		exit(EXIT_FAILURE);
    }

	std::vector<cv::Point> pl;
	std::cout<<"hey dear, am I here.."<<std::endl;
	while (!infile.eof())
	{
		cv::Point point;
		infile >> point.y;
		std::cout<<"point.x is...."<<point.x<<std::endl;
		infile >> point.x;
		std::cout<<"point.y is...."<<point.y<<std::endl;
		pl.push_back(point);
	}

	infile.close();

	return pl;
}

cv::Mat readImage(const char* fileName) 
{
	cv::Mat image;
	//image = cvLoadImage("C:\Users\I843001\Desktop\lena512color.jpg");
	image = cv::imread(fileName, CV_LOAD_IMAGE_ANYDEPTH);

	return image;
}

void writeImage(cv::Mat img,const char* fileName)
{
	cv::imwrite(fileName, img);
}