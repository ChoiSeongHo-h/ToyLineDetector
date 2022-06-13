#include <iostream>
#include "opencv2/opencv.hpp"
#include <limits>

enum Consts
{
	ORIGINAL_FRAME_H = 240,
	ORIGINAL_FRAME_W = 320,
	ROI_H = 65,
	GAUSSIANBLUR_SIZE = 7,
	ADAPTIVETHRESHOLD_SIZE = 11,
	ADAPTIVETHRESHOLD_C = 2,
	MEDIANBLUR_SIZE = 9,
	CANNY_THRESHOLD_1ST = 180,
	CANNY_THRESHOLD_2ND = 360,
	HOUGHLINESP_RHO = 1,
	HOUGHLINESP_THRESHOLD = 30,
	HOUGHLINESP_MIN_LINE_LEN = 12,
	HOUGHLINESP_MAX_LINE_GAP = 15,
};

int main()
{
	cv::VideoCapture cam(0, cv::CAP_DSHOW);
	if (!cam.isOpened())
	{
		std::cerr << "The camera cannot be opened." << std::endl;
		return -1;
	}
	cam.set(cv::CAP_PROP_FRAME_WIDTH, ORIGINAL_FRAME_W);
	cam.set(cv::CAP_PROP_FRAME_HEIGHT, ORIGINAL_FRAME_H);

	while(1)
	{
		cv::Mat originFrame;
		cam.read(originFrame);
		if (originFrame.empty())
			std::cerr << "Capture Failed" << std::endl;

		cv::Mat grayFrame;
		cv::cvtColor(originFrame, grayFrame, cv::COLOR_BGR2GRAY);

		//preprocessing
		cv::Mat roiFrame = grayFrame(cv::Rect(0, ORIGINAL_FRAME_H - ROI_H - 1, ORIGINAL_FRAME_W, ROI_H));
		cv::GaussianBlur(roiFrame, roiFrame, cv::Size(GAUSSIANBLUR_SIZE, GAUSSIANBLUR_SIZE), 0);
		cv::adaptiveThreshold(roiFrame, roiFrame, UCHAR_MAX, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, ADAPTIVETHRESHOLD_SIZE, ADAPTIVETHRESHOLD_C);
		cv::medianBlur(roiFrame, roiFrame, MEDIANBLUR_SIZE);
		cv::Canny(roiFrame, roiFrame, CANNY_THRESHOLD_1ST, CANNY_THRESHOLD_2ND);

		//get line
		std::vector<cv::Vec4i> detectedLines;
		cv::HoughLinesP(roiFrame, detectedLines, HOUGHLINESP_RHO, CV_PI / 180, HOUGHLINESP_THRESHOLD, HOUGHLINESP_MIN_LINE_LEN, HOUGHLINESP_MAX_LINE_GAP);

		double vanishingPointY = -std::numeric_limits<double>::infinity();
		double linesCos = -std::numeric_limits<double>::infinity();
		int crossArgi = -1;
		int crossArgj = -1;
		int parallelArgi = -1;
		int parallelArgj = -1;

		//find good line (cross or parallel)
		for (int i = 0; i < detectedLines.size() - 1; i++)
		{
			for (int j = i; j < detectedLines.size(); j++)
			{
				const int& l0X0 = detectedLines[i][0];
				const int& l0Y0 = detectedLines[i][1];
				const int& l0X1 = detectedLines[i][2];
				const int& l0Y1 = detectedLines[i][3];

				const int& l1X0 = detectedLines[j][0];
				const int& l1Y0 = detectedLines[j][1];
				const int& l1X1 = detectedLines[j][2];
				const int& l1Y1 = detectedLines[j][3];

				double vanishingPointYDenominator = (l0X0 - l0X1)*(l1Y0 - l1Y1) - (l0Y0 - l0Y1)*(l1X0 - l1X1);

				if (vanishingPointYDenominator)
				{
					int l0W = l0X0 - l0X1;
					int l0H = l0Y0 - l0Y1;
					int l1W = l1X0 - l1X1;
					int l1H = l1Y0 - l1Y1;
					double temLinesCos = (l0W * l1W + l0H * l1H) / (sqrt(pow(l0W, 2) + pow(l0H, 2)) * sqrt(pow(l1W, 2) + pow(l1H, 2)));
					double tempVanishingPointY = ((l0X0 * l0Y1 - l0Y0 * l0X1) * (l1Y0 - l1Y1) - (l0Y0 - l0Y1) * (l1X0 * l1Y1 - l1Y0 * l1X1)) / vanishingPointYDenominator;

					//find cross / parallel
					if (tempVanishingPointY<0 && tempVanishingPointY>vanishingPointY && temLinesCos < 0.94)
					{
						vanishingPointY = tempVanishingPointY;
						crossArgi = i;
						crossArgj = j;
					}
					else if (linesCos < temLinesCos)
					{
						linesCos = temLinesCos;
						parallelArgi = i;
						parallelArgj = j;
					}
				}
			}
		}

		short controlOutput = 0;
		cv::Mat linedROI;
		cvtColor(roiFrame, linedROI, cv::COLOR_GRAY2BGR);
		//output for cross / parallel
		if (crossArgi != -1)
		{
			cv::line(linedROI, cv::Point(detectedLines[crossArgi][0], detectedLines[crossArgi][1]), cv::Point(detectedLines[crossArgi][2], detectedLines[crossArgi][3]), cv::Scalar(0, 0, 255), 1);
			line(linedROI, cv::Point(detectedLines[crossArgj][0], detectedLines[crossArgj][1]), cv::Point(detectedLines[crossArgj][2], detectedLines[crossArgj][3]), cv::Scalar(0, 0, 255), 1);
			circle(linedROI, cv::Point(((detectedLines[crossArgi][0] * detectedLines[crossArgi][3] - detectedLines[crossArgi][1] * detectedLines[crossArgi][2]) * (detectedLines[crossArgj][0] - detectedLines[crossArgj][2]) - (detectedLines[crossArgi][0] - detectedLines[crossArgi][2]) * (detectedLines[crossArgj][0] * detectedLines[crossArgj][3] - detectedLines[crossArgj][1] * detectedLines[crossArgj][2])) / ((detectedLines[crossArgi][0] - detectedLines[crossArgi][2]) * (detectedLines[crossArgj][1] - detectedLines[crossArgj][3]) - (detectedLines[crossArgi][1] - detectedLines[crossArgi][3]) * (detectedLines[crossArgj][0] - detectedLines[crossArgj][2])), 0), 5, cv::Scalar(255, 0, 0));
			controlOutput = ((detectedLines[crossArgi][0] * detectedLines[crossArgi][3] - detectedLines[crossArgi][1] * detectedLines[crossArgi][2]) * (detectedLines[crossArgj][0] - detectedLines[crossArgj][2]) - (detectedLines[crossArgi][0] - detectedLines[crossArgi][2]) * (detectedLines[crossArgj][0] * detectedLines[crossArgj][3] - detectedLines[crossArgj][1] * detectedLines[crossArgj][2])) / ((detectedLines[crossArgi][0] - detectedLines[crossArgi][2]) * (detectedLines[crossArgj][1] - detectedLines[crossArgj][3]) - (detectedLines[crossArgi][1] - detectedLines[crossArgi][3]) * (detectedLines[crossArgj][0] - detectedLines[crossArgj][2])) - 160;
		}
		else if (parallelArgi != -1)
		{
			line(linedROI, cv::Point(detectedLines[parallelArgi][0], detectedLines[parallelArgi][1]), cv::Point(detectedLines[parallelArgi][2], detectedLines[parallelArgi][3]), cv::Scalar(0, 0, 255), 1);
			line(linedROI, cv::Point(detectedLines[parallelArgj][0], detectedLines[parallelArgj][1]), cv::Point(detectedLines[parallelArgj][2], detectedLines[parallelArgj][3]), cv::Scalar(0, 0, 255), 1);

			if (detectedLines[parallelArgi][0] - detectedLines[parallelArgi][2])
			{
				if (atan(double(detectedLines[parallelArgi][1] - detectedLines[parallelArgi][3]) / float(detectedLines[parallelArgi][0] - detectedLines[parallelArgi][2])) * 180 / CV_PI > 0)
					controlOutput = -160;
				else
					controlOutput = 160;
			}
		}

		controlOutput = int((float(controlOutput) / 160 + 1) / 2 * 255);
		std::cout << controlOutput << std::endl;
		imshow("dst", linedROI);
		cv::waitKey(25);
	}
}