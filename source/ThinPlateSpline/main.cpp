#include<iostream>
#include<fstream>

#include <stdlib.h> 

# include "CThinPlateSpline.h"


std::vector<cv::Point> readPList(std::string);
cv::Mat readImage(std::string);
void writeImage(cv::Mat,std::string);

int main(int argc, char **argv)
{
	if( argc < 4)
    {
		std::cout <<" Not enought parameters, please specify pS, pD, src" << std::endl;
		return -1;
    }

	std::vector<cv::Point> pS;
	std::vector<cv::Point> pD;
	pS = readPList(argv[1]);
	pD = readPList(argv[2]);

	cv::Mat img = readImage(argv[3]);

	if( argc >= 4)
		writeImage(cv::Mat,std::string);

	return 0;
}

std::vector<cv::Point> readPList(std::string fileName)
{
	std::ifstream infile(fileName);
	if (! infile) 
	{
		std::cerr << "unable to open the file  for reading" << std::endl;
		exit(EXIT_FAILURE);
    }

	std::vector<cv::Point> pl;
	while (!infile.eof())
	{
		cv::Point point;
		infile >> point.x;
		infile >> point.y;
		pl.push_back(point);
	}

	infile.close();

	return pl;
}

cv::Mat readImage(std::string)
{
	cv::Mat image;
	image = cv::imread(string, cv::CV_LOAD_IMAGE_GRAYSCALE);

	return image
}

void writeImage(cv::Mat img,std::string fileName)
{
	cv::imwrite(fileName, img);
}