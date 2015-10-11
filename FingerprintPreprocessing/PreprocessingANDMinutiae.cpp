#include "stdafx.h"
#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include "opencv/highgui.h"
#include "stdio.h"
#include "opencv2\imgproc\types_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2\highgui\highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <opencv2\core\core.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

const int maxCycle = 10;
const int maxPoints = 500;
const int WSIZE = 12;

CvRect erasedPixels[maxCycle][maxPoints];
int nrErased[maxCycle];

CvRect thorns[maxPoints][maxCycle];
int thornLength[maxPoints] = { 0 };
int nrThorns;

//Minutiae
//--------
struct Minutia
{
	int event; // 1 - elagazas, 2 - vegzodes
	double x;
	double y;
};

//Set Value
//---------
void SetValue(IplImage* im, int x, int y, bool value)
{
	im->imageData[y*im->widthStep + x] = (unsigned char)(value ? 0 : 255);
}

//Get Value
//---------
bool GetValue(IplImage* im, int x, int y)
{
	return ((unsigned char)im->imageData[y*im->widthStep + x] < 128);
}

//Set Color
//---------
void setColor(IplImage* im, int x, int y, unsigned char value)
{
	im->imageData[y*im->widthStep + x] = value;
}

//Get Color
//---------
unsigned char getColor(IplImage* im, int x, int y)
{
	return (unsigned char)im->imageData[y*im->widthStep + x];
}

//Set RGB
//-------
void setRGB(IplImage* im, int x, int y, unsigned char r, unsigned char g, unsigned char b)
{
	im->imageData[y*im->widthStep + 3 * x + 0] = b;
	im->imageData[y*im->widthStep + 3 * x + 1] = g;
	im->imageData[y*im->widthStep + 3 * x + 2] = r;
}

//Get RGB
//-------
void getRGB(IplImage* im, int x, int y, unsigned char& r, unsigned char& g, unsigned char& b)
{
	b = im->imageData[y*im->widthStep + 3 * x + 0];
	g = im->imageData[y*im->widthStep + 3 * x + 1];
	r = im->imageData[y*im->widthStep + 3 * x + 2];
}

//Mark Blue
//---------
bool isBlue(IplImage* im, int x, int y)
{
	unsigned char r, g, b;
	getRGB(im, x, y, r, g, b);
	return (b == 255 && r == 0 && g == 0);
}

//Mark Red
//--------
bool isRed(IplImage* im, int x, int y)
{
	unsigned char r, g, b;
	getRGB(im, x, y, r, g, b);
	return (r == 255 && g == 0 && b == 0);
}

//Mark Green
//----------
bool isGreen(IplImage* im, int x, int y)
{
	unsigned char r, g, b;
	getRGB(im, x, y, r, g, b);
	return (g == 255 && r == 0 && b == 0);
}

//Mark Gray
//---------
bool isGray(IplImage* im, int x, int y)
{
	unsigned char r, g, b;
	int i, j;
	i = (x / WSIZE) * WSIZE + WSIZE / 2;
	j = (y / WSIZE) * WSIZE + WSIZE / 2;
	getRGB(im, i, j, r, g, b);
	return (g == 192 && r == 192 && b == 192);
}

//Need to thinning the fingerprint image
//--------------------------------------
int GolayL(IplImage* src, IplImage* dst)
{
	int count;
	int i = 0;
	do
	{
		count = 0;
		for (int index = 1; index <= 8; index++)
		{
			for (int x = 1; x<src->width - 1; x++)
			for (int y = 1; y<src->height - 1; y++)
			if (GetValue(src, x, y))
			{
				switch (index){
				case 1:
					if (!GetValue(src, x - 1, y - 1) && !GetValue(src, x, y - 1) && !GetValue(src, x + 1, y - 1) &&
						GetValue(src, x, y) &&
						GetValue(src, x - 1, y + 1) && GetValue(src, x, y + 1) && GetValue(src, x + 1, y + 1))
					{
						SetValue(dst, x, y, false);
						count++;
					}
					break;
				case 2:
					if (!GetValue(src, x, y - 1) && !GetValue(src, x + 1, y - 1) &&
						GetValue(src, x - 1, y) && GetValue(src, x, y) && !GetValue(src, x + 1, y) &&
						GetValue(src, x, y + 1))
					{
						SetValue(dst, x, y, false);
						count++;
					}
					break;
				case 3:
					if (GetValue(src, x - 1, y - 1) && !GetValue(src, x + 1, y - 1) &&
						GetValue(src, x - 1, y) && GetValue(src, x, y) && !GetValue(src, x + 1, y) &&
						GetValue(src, x - 1, y + 1) && !GetValue(src, x + 1, y + 1))
					{
						SetValue(dst, x, y, false);
						count++;
					}
					break;
				case 4:
					if (GetValue(src, x, y - 1) &&
						GetValue(src, x - 1, y) && GetValue(src, x, y) && !GetValue(src, x + 1, y) &&
						!GetValue(src, x, y + 1) && !GetValue(src, x + 1, y + 1))
					{
						SetValue(dst, x, y, false);
						count++;
					}
					break;
				case 5:
					if (GetValue(src, x - 1, y - 1) && GetValue(src, x, y - 1) && GetValue(src, x + 1, y - 1) &&
						GetValue(src, x, y) &&
						!GetValue(src, x - 1, y + 1) && !GetValue(src, x, y + 1) && !GetValue(src, x + 1, y + 1))
					{
						SetValue(dst, x, y, false);
						count++;
					}
					break;
				case 6:
					if (GetValue(src, x, y - 1) &&
						!GetValue(src, x - 1, y) && GetValue(src, x, y) && GetValue(src, x + 1, y) &&
						!GetValue(src, x - 1, y + 1) && !GetValue(src, x, y + 1))
					{
						SetValue(dst, x, y, false);
						count++;
					}
					break;
				case 7:
					if (!GetValue(src, x - 1, y - 1) && GetValue(src, x + 1, y - 1) &&
						!GetValue(src, x - 1, y) && GetValue(src, x, y) && GetValue(src, x + 1, y) &&
						!GetValue(src, x - 1, y + 1) && GetValue(src, x + 1, y + 1))
					{
						SetValue(dst, x, y, false);
						count++;
					}
					break;
				case 8:
					if (!GetValue(src, x - 1, y - 1) && !GetValue(src, x, y - 1) &&
						!GetValue(src, x - 1, y) && GetValue(src, x, y) && GetValue(src, x + 1, y) &&
						GetValue(src, x, y + 1))
					{
						SetValue(dst, x, y, false);
						count++;
					}
					break;
				}
			}
			cvResize(dst, src);
		}
		printf("i = %d, count = %d\n", i, count);
		cvShowImage("The Fingerprint", src);

		cvWaitKey(300);
		i++;
	} while (count>0);
	return i;
}

//Need to find the ridge ending in fingerprint
//--------------------------------------------
int GolayE(IplImage* im, IplImage* im2)
{
	int count;
	int i = 0;
	int cycle = 0;
	do
	{
		count = 0;
		for (int index = 1; index <= 8; index++)
		{
			for (int x = 1; x<im->width - 1; x++)
			for (int y = 1; y<im->height - 1; y++)
			if (GetValue(im, x, y))
			{
				switch (index){
				case 1:
					if (GetValue(im, x, y) && GetValue(im, x, y - 1) &&
						!GetValue(im, x - 1, y) && !GetValue(im, x + 1, y) &&
						!GetValue(im, x - 1, y + 1) && !GetValue(im, x, y + 1) && !GetValue(im, x + 1, y + 1)
						&& GetValue(im2, x, y))
					{
						SetValue(im2, x, y, false);
						erasedPixels[cycle][count++] = cvRect(x, y, 2, 1);
					}
					break;
				case 2:
					if (!GetValue(im, x - 1, y - 1) &&
						!GetValue(im, x - 1, y) && GetValue(im, x, y) && !GetValue(im, x + 1, y) &&
						!GetValue(im, x - 1, y + 1) && !GetValue(im, x, y + 1) && !GetValue(im, x + 1, y + 1)
						&& GetValue(im2, x, y))
					{
						SetValue(im2, x, y, false);
						erasedPixels[cycle][count++] = cvRect(x, y, 2, 1);
					}
					break;
				case 3:
					if (!GetValue(im, x - 1, y - 1) && !GetValue(im, x, y - 1) &&
						!GetValue(im, x - 1, y) && GetValue(im, x, y) && GetValue(im, x + 1, y) &&
						!GetValue(im, x - 1, y + 1) && !GetValue(im, x, y + 1)
						&& GetValue(im2, x, y))
					{
						SetValue(im2, x, y, false);
						erasedPixels[cycle][count++] = cvRect(x, y, 2, 1);
					}
					break;
				case 4:
					if (!GetValue(im, x - 1, y - 1) && !GetValue(im, x, y - 1) && !GetValue(im, x + 1, y - 1) &&
						!GetValue(im, x - 1, y) && GetValue(im, x, y) &&
						!GetValue(im, x - 1, y + 1) && !GetValue(im, x, y + 1)
						&& GetValue(im2, x, y))
					{
						SetValue(im2, x, y, false);
						erasedPixels[cycle][count++] = cvRect(x, y, 2, 1);
					}
					break;
				case 5:
					if (!GetValue(im, x - 1, y - 1) && !GetValue(im, x, y - 1) && !GetValue(im, x + 1, y - 1) &&
						!GetValue(im, x - 1, y) && GetValue(im, x, y) && !GetValue(im, x + 1, y) &&
						GetValue(im, x, y + 1)
						&& GetValue(im2, x, y))
					{
						SetValue(im2, x, y, false);
						erasedPixels[cycle][count++] = cvRect(x, y, 2, 1);
					}
					break;
				case 6:
					if (!GetValue(im, x - 1, y - 1) && !GetValue(im, x, y - 1) && !GetValue(im, x + 1, y - 1) &&
						!GetValue(im, x - 1, y) && GetValue(im, x, y) && !GetValue(im, x + 1, y) &&
						!GetValue(im, x + 1, y + 1)
						&& GetValue(im2, x, y))
					{
						SetValue(im2, x, y, false);
						erasedPixels[cycle][count++] = cvRect(x, y, 2, 1);
					}
					break;
				case 7:
					if (!GetValue(im, x, y - 1) && !GetValue(im, x + 1, y - 1) &&
						GetValue(im, x - 1, y) && GetValue(im, x, y) && !GetValue(im, x + 1, y) &&
						!GetValue(im, x, y + 1) && !GetValue(im, x + 1, y + 1)
						&& GetValue(im2, x, y))
					{
						SetValue(im2, x, y, false);
						erasedPixels[cycle][count++] = cvRect(x, y, 2, 1);
					}
					break;
				case 8:
					if (!GetValue(im, x, y - 1) && !GetValue(im, x + 1, y - 1) &&
						GetValue(im, x, y) && !GetValue(im, x + 1, y) &&
						!GetValue(im, x - 1, y + 1) && !GetValue(im, x, y + 1) && !GetValue(im, x + 1, y + 1)
						&& GetValue(im2, x, y))
					{
						SetValue(im2, x, y, false);
						erasedPixels[cycle][count++] = cvRect(x, y, 2, 1);
					}
					break;
				}
			}
		}
		cvCopy(im2, im);
		printf("i = %d, count = %d\n", i, count);
		cvShowImage("The Fingerprint", im);
		nrErased[cycle] = count;

		cvWaitKey(300);
		i++;
		cycle++;

	} while (count>0 && cycle<maxCycle);
	return i;
}

//Need to find the ridge bifurcation in fingerprint
//-------------------------------------------------
bool GolayQcond(IplImage* src, int x, int y, int index)
{
	bool res;
	switch (index){
	case 1:
		res = GetValue(src, x - 1, y - 1) && !GetValue(src, x, y - 1) && GetValue(src, x + 1, y - 1) &&
			GetValue(src, x, y) &&
			GetValue(src, x - 1, y + 1) && !GetValue(src, x, y + 1);
		break;
	case 2:
		res = GetValue(src, x - 1, y - 1) &&
			!GetValue(src, x - 1, y) && GetValue(src, x, y) && !GetValue(src, x + 1, y) &&
			GetValue(src, x - 1, y + 1) && GetValue(src, x + 1, y + 1);
		break;
	case 3:
		res = !GetValue(src, x, y - 1) && GetValue(src, x + 1, y - 1) &&
			GetValue(src, x, y) &&
			GetValue(src, x - 1, y + 1) && !GetValue(src, x, y + 1) && GetValue(src, x + 1, y + 1);
		break;
	case 4:
		res = GetValue(src, x - 1, y - 1) && GetValue(src, x + 1, y - 1) &&
			!GetValue(src, x - 1, y) && GetValue(src, x, y) && !GetValue(src, x + 1, y) &&
			GetValue(src, x + 1, y + 1);
		break;
	case 5:
		res = GetValue(src, x - 1, y + 1) && !GetValue(src, x, y + 1) && GetValue(src, x + 1, y + 1) &&
			GetValue(src, x, y) &&
			GetValue(src, x - 1, y - 1) && !GetValue(src, x, y - 1);
		break;
	case 6:
		res = GetValue(src, x + 1, y - 1) &&
			!GetValue(src, x - 1, y) && GetValue(src, x, y) && !GetValue(src, x + 1, y) &&
			GetValue(src, x - 1, y + 1) && GetValue(src, x + 1, y + 1);
		break;
	case 7:
		res = !GetValue(src, x, y + 1) && GetValue(src, x + 1, y + 1) &&
			GetValue(src, x, y) &&
			GetValue(src, x - 1, y - 1) && !GetValue(src, x, y - 1) && GetValue(src, x + 1, y - 1);
		break;
	case 8:
		res = GetValue(src, x - 1, y - 1) && GetValue(src, x + 1, y - 1) &&
			!GetValue(src, x - 1, y) && GetValue(src, x, y) && !GetValue(src, x + 1, y) &&
			GetValue(src, x - 1, y + 1);
		break;
	case 9:
		res = GetValue(src, x - 1, y - 1) && !GetValue(src, x, y - 1) && !GetValue(src, x + 1, y - 1) &&
			!GetValue(src, x - 1, y) && GetValue(src, x, y) && GetValue(src, x + 1, y) &&
			!GetValue(src, x - 1, y + 1) && GetValue(src, x, y + 1) && !GetValue(src, x + 1, y + 1);
		break;
	case 10:
		res = !GetValue(src, x - 1, y - 1) && GetValue(src, x, y - 1) && !GetValue(src, x + 1, y - 1) &&
			!GetValue(src, x - 1, y) && GetValue(src, x, y) && GetValue(src, x + 1, y) &&
			GetValue(src, x - 1, y + 1) && !GetValue(src, x, y + 1) && !GetValue(src, x + 1, y + 1);
		break;
	case 11:
		res = !GetValue(src, x - 1, y - 1) && GetValue(src, x, y - 1) && !GetValue(src, x + 1, y - 1) &&
			GetValue(src, x - 1, y) && GetValue(src, x, y) && !GetValue(src, x + 1, y) &&
			!GetValue(src, x - 1, y + 1) && !GetValue(src, x, y + 1) && GetValue(src, x + 1, y + 1);
		break;
	case 12:
		res = !GetValue(src, x - 1, y - 1) && !GetValue(src, x, y - 1) && GetValue(src, x + 1, y - 1) &&
			GetValue(src, x - 1, y) && GetValue(src, x, y) && !GetValue(src, x + 1, y) &&
			!GetValue(src, x - 1, y + 1) && GetValue(src, x, y + 1) && !GetValue(src, x + 1, y + 1);
		break;
	case 13:
		res = !GetValue(src, x, y - 2) &&
			GetValue(src, x - 1, y - 1) && GetValue(src, x + 1, y - 1) &&
			!GetValue(src, x - 1, y) && GetValue(src, x, y) && !GetValue(src, x + 1, y) &&
			!GetValue(src, x - 1, y + 1) && GetValue(src, x, y + 1) && !GetValue(src, x + 1, y + 1);
		break;
	case 14:
		res = GetValue(src, x - 1, y - 1) && !GetValue(src, x, y - 1) && !GetValue(src, x + 1, y - 1) &&
			!GetValue(src, x - 2, y) && GetValue(src, x, y) && GetValue(src, x + 1, y) &&
			GetValue(src, x - 1, y + 1) && !GetValue(src, x, y + 1) && !GetValue(src, x + 1, y + 1);
		break;
	case 15:
		res = !GetValue(src, x - 1, y - 1) && GetValue(src, x, y - 1) && !GetValue(src, x + 1, y - 1) &&
			!GetValue(src, x - 1, y) && GetValue(src, x, y) && !GetValue(src, x + 1, y) &&
			GetValue(src, x - 1, y + 1) && GetValue(src, x + 1, y + 1) &&
			!GetValue(src, x, y + 2);

		break;
	case 16:
		res = !GetValue(src, x - 1, y - 1) && !GetValue(src, x, y - 1) && GetValue(src, x + 1, y - 1) &&
			GetValue(src, x - 1, y) && GetValue(src, x, y) && !GetValue(src, x + 2, y) &&
			!GetValue(src, x - 1, y + 1) && !GetValue(src, x, y + 1) && GetValue(src, x + 1, y + 1);
		break;
	case 17:
		res = !GetValue(src, x - 1, y - 1) && GetValue(src, x, y - 1) && !GetValue(src, x + 1, y - 1) &&
			GetValue(src, x - 1, y) && GetValue(src, x, y) && GetValue(src, x + 1, y) &&
			!GetValue(src, x - 1, y + 1) && GetValue(src, x, y + 1) && !GetValue(src, x + 1, y + 1);
		break;
	}
	return res;
}

//Need to find the ridge bifurcation in fingerprint
//-------------------------------------------------
int GolayQ(IplImage* src, IplImage* dst)
{
	int count = 0;

	for (int x = 2; x<src->width - 2; x++)
	for (int y = 2; y<src->height - 2; y++)
	if (GetValue(src, x, y))
	{
		for (int index = 1; index <= 17; index++)
		{
			if (GolayQcond(src, x, y, index))
			{
				setRGB(dst, x, y, 255, 0, 0);
				count++;
			}
		}
	}

	return count;
}

int abso(int a)
{
	return a<0 ? -a : a;
}




int main()
{
	
	//Open and Read the Image
	//----------------------- 
	cv::Mat imInput = cv::imread("fingerprint2.bmp", CV_LOAD_IMAGE_COLOR);
	cv::imshow("The Fingerprint",imInput);
	cv::waitKey();

	//Remove Noise by Blurring with a Gaussian Filter
	//-----------------------------------------------
	cv::Mat img_filter;
	cv::GaussianBlur(imInput, img_filter, Size(3, 3), 0, 0, BORDER_DEFAULT);
	
	//Grayscale matrix
	//----------------
	cv::Mat grayscaleMat(img_filter.size(), CV_8U);

	//Convert BGR to Gray
	//-------------------
	cv::cvtColor(img_filter, grayscaleMat, CV_BGR2GRAY);

	//Equalize the histogram
	//----------------------
	cv::Mat img_hist_equalized;
	cv::equalizeHist(grayscaleMat, img_hist_equalized);

	//Binary image
	//------------
	cv::Mat binaryMat(img_hist_equalized.size(), img_hist_equalized.type());

	//Apply thresholding
	//------------------
	cv::threshold(img_hist_equalized, binaryMat, 100, 255, cv::THRESH_BINARY);

	//Save image
	//----------
	cv::imwrite("binary.bmp", binaryMat);
	
	//Load the saved binary image
	//---------------------------
	IplImage *im_binary_input, *im_binary_input_clone, *im_clean;
	im_binary_input = cvLoadImage("binary.bmp", 0);

	//Clone and show the saved binary image
	//-------------------------------------
	im_binary_input_clone = cvCloneImage(im_binary_input);
	cvShowImage("The Fingerprint", im_binary_input);

	//Thinning the fingerprint image with Golay ABC L mask algorithm
	//--------------------------------------------------------------
	GolayL(im_binary_input, im_binary_input_clone);

	//Save the thinned image
	//-----------------------
	cvSaveImage("thinning.bmp", im_binary_input_clone);

	//The im_binary_input image will be the thinned image
	//----------------------------------------------------
	cvCopy(im_binary_input_clone, im_binary_input);

	//Find the ridge ending in fingerprint with Golay ABC E mask algorithm
	//--------------------------------------------------------------------
	GolayE(im_binary_input, im_binary_input_clone);

	//Copy the im_binary_input_clone image in the im_binary_input image
	//-----------------------------------------------------------------
	cvCopy(im_binary_input_clone, im_binary_input);

	//Make a ROI on the im_binary_input image and on the im_binary_input_clone image
	//------------------------------------------------------------------------------
	cvSet(im_binary_input_clone, cvScalar(255, 0, 0, 0));
	CvRect roi = cvRect(2, 2, im_binary_input_clone->width - 4, im_binary_input_clone->height - 4);
	cvSetImageROI(im_binary_input, roi);
	cvSetImageROI(im_binary_input_clone, roi);
	cvCopy(im_binary_input, im_binary_input_clone);
	cvResetImageROI(im_binary_input);
	cvResetImageROI(im_binary_input_clone);

	//Delete the thorns from to the binary fingerprint image
	//------------------------------------------------------
	int i, j, k, l, count;

	for (i = 0; i<nrErased[0]; i++)
	{
		thorns[i][0] = erasedPixels[0][i];
		thornLength[i] = 1;
	}

	nrThorns = nrErased[0];

	for (k = 1; k<maxCycle; k++)
	{
		count = 0;
		for (j = 0; j<nrErased[k]; j++)
		{
			bool found = false;
			for (i = 0; i<nrThorns; i++)
			{
				if (abso(erasedPixels[k][j].x - thorns[i][k - 1].x) <= 1 && abso(erasedPixels[k][j].y - thorns[i][k - 1].y) <= 1)
				{
					thorns[i][k] = erasedPixels[k][j];
					thornLength[i]++;
					found = true;
				}
			}
			if (!found) count++;
		}
		printf("Thorn report: %d %d\n", k, count);
	}

	for (i = 0; i<nrThorns; i++) if (thornLength[i] >= 5)
	{
		for (j = 0; j<thornLength[i]; j++)
		{
			SetValue(im_binary_input_clone, thorns[i][j].x, thorns[i][j].y, true);
		}
	}

	cvShowImage("The Fingerprint", im_binary_input_clone);
	cvSaveImage("clean.bmp", im_binary_input_clone);

	//Open the anti thorn fingerprint image
	//-------------------------------------
	im_clean = cvLoadImage("clean.bmp", 1);

	//
	//
	for (i = 2; i<im_binary_input_clone->width - 2; i++) 
	for (j = 2; j<im_binary_input_clone->height - 2; j++)
	if (GetValue(im_binary_input_clone, i, j))
	{
		count = 0;
		for (k = i - 1; k <= i + 1; k++) 
		for (l = j - 1; l <= j + 1; l++)
		if ((k != i || l != j) && GetValue(im_binary_input_clone, k, l)) 
			count++;
		if (count == 1) 
			setRGB(im_clean, i, j, 0, 255, 0);
	}

	//Find the ridge bifurcation in fingerprint
	//-----------------------------------------
	GolayQ(im_binary_input_clone, im_clean);

	//Mark the minutiae points
	//(Red -> Ridge Bifurcation);(Green -> Ridge Ending);(Blue ->Margin of Fingerprint)
	//---------------------------------------------------------------------------------
	Minutia minu;

	FILE* f;
	f = fopen("minutiae.min", "wb+");

	for (i = 2; i<im_clean->width - 2; i++) 
	for (j = 2; j<im_clean->height - 2; j++)
	if (isRed(im_clean, i, j) || isGreen(im_clean, i, j))
	{
		bool kill = false;

		if (i<WSIZE || i >= im_clean->width - WSIZE) 
			kill = true;
		if (j<WSIZE || j >= im_clean->height - WSIZE) 
			kill = true;
		
		if (!kill)
		{
			if (isGray(im_clean, i + WSIZE, j))	
				kill = true;
			if (isGray(im_clean, i - WSIZE, j))	
				kill = true;
			if (isGray(im_clean, i, j + WSIZE))	
				kill = true;
			if (isGray(im_clean, i, j - WSIZE))	
				kill = true;
		}

		if (kill) 
			setRGB(im_clean, i, j, 0, 0, 255);
		else
		{
			minu.event = isRed(im_clean, i, j) ? 1 : 2;
			minu.x = (double)i;
			minu.y = (double)j;
			fwrite(&minu, 1, sizeof(minu), f);
		}
	}

	fclose(f);
	cvSaveImage("minutiae.bmp", im_clean);
	cvShowImage("The Fingerprint", im_clean);


	cv::waitKey();
	return 0;
	
}