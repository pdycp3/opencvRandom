#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;
void readme();

/** @function main */
int main(int argc, char** argv)
{
	Mat img1 = imread("D://data//south-building//images//P1180141.JPG", IMREAD_GRAYSCALE);
	Mat img2 = imread("D://data//south-building//images//P1180142.JPG", IMREAD_GRAYSCALE);

	if (!img1.data || !img2.data)
	{
		cout << "Error reading images!!" << endl;
		return -1;
	}

	vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptor_1, descriptor_2;
	Ptr<SIFT> siftFloat1 = cv::SIFT::create(0, 3, 0.04, 10, 1.6);
	siftFloat1->detectAndCompute(img1, Mat(), keypoints_1, descriptor_1, false);


	Ptr<SIFT> siftFloat2 = cv::SIFT::create(0, 3, 0.04, 10, 1.6);
	siftFloat2->detectAndCompute(img2, Mat(), keypoints_2, descriptor_2, false);

	// 	SiftFeatureDetector detector;
	// 	vector<KeyPoint> keypoints1, keypoints2;
	// 	detector.create();
	// 	detector.detect(img1, keypoints1);
	// 	detector.detect(img2, keypoints2);
	// 
	// 	SiftDescriptorExtractor extractor;
	// 	Mat descriptor1, descriptor2;
	// 	extractor.compute(img1, keypoints1, descriptor1);
	// 	extractor.compute(img2, keypoints2, descriptor2);

		//FlannBasedMatcher matcher;
	FlannBasedMatcher matcher;
	vector<DMatch> matches;
	matcher.match(descriptor_1, descriptor_2, matches, Mat());
	






	//-- Step 3: Matching descriptor vectors using FLANN matcher


	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptor_1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
	//-- PS.- radiusMatch can also be used here.
	std::vector< DMatch > good_matches;

	for (int i = 0; i < descriptor_1.rows; i++)
	{
		if (matches[i].distance < 2 * min_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}

	//-- Draw only "good" matches
	Mat img_matches;
	drawMatches(img1, keypoints_1, img2, keypoints_2,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//-- Show detected matches
	imshow("Good Matches", img_matches);

	for (int i = 0; i < good_matches.size(); i++)
	{
		printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx);
	}
	Mat imgmatches;
	drawMatches(img1, keypoints_1, img2, keypoints_2, matches, imgmatches, Scalar::all(-1), Scalar::all(-1));
	imwrite("D:\\data\\south-building\\images\\contact.JPG", img_matches);
	//imshow("Matches", imgmatches);


	waitKey(0);

	return 0;
}

/** @function readme */
void readme()
{
	std::cout << " Usage: ./SURF_FlannMatcher <img1> <img2>" << std::endl;
}
