#include <stdio.h>
#include <time.h>
#include <iostream>
#include <opencv2/core/utility.hpp>
#include "opencv2/core/core.hpp"
#include <opencv2/core/types_c.h>
#include "opencv2/features2d/features2d.hpp"

#include "opencv2/highgui/highgui.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
using namespace cv;
using namespace std;
class myTime {

	char* name;
	double begin;
	double end;
public:
	myTime( char*pName) {
		name = new char[strlen(pName) + 1];
		cout << "Create clock" << pName << "starting..." << endl;
		strcpy(name, pName);							//��ʼ���������
		begin = end = 0.0;					    	//��ʼ�����ݳ�Աbegin��end

	};
	myTime(clock_t t, char* pName) {
		name = new char[strlen(pName) + 1];
		cout << "Create objective" << pName << "starting..." << endl;
		strcpy(name, pName);							//��ʼ���������
		begin = (double)t / CLOCKS_PER_SEC;
		end = 0.0;
	};
	~myTime() {
		cout << "destruct object" << name << endl;
		delete[] name;
	};

	void start() {
		begin = (double)clock() / CLOCKS_PER_SEC;
	};
	void stop() {
		end = (double)clock() / CLOCKS_PER_SEC;
		show();
	};
	void show() {
		cout << "clock name is" << name << endl;
		cout << "start��" << begin << "second" << endl;
		cout << "end��" << end << "second" << endl;
		cout << "timecost��" << (end - begin) << "��" << endl;
	};
};
//ʮ����ͳ�ƺ���
void RunExactFeatureTimeTest(string testType,cv::Mat & imagetest)
{

	Ptr<FeatureDetector> ptrFeatureExact;
	string  strSift = "SIFT";
	if (testType==strSift)
	{
		ptrFeatureExact = cv::SIFT::create(0, 3, 0.04, 10, 1.6);
		cout << "test SIFT............" << endl;
	}

	else if (testType == string("ORB"))
	{
		ptrFeatureExact = cv::ORB::create();
		cout << "test ORB.............." << endl;
	}
	else if (testType == string("BRISK"))
	{
		ptrFeatureExact = cv::BRISK::create();
		cout << "test BRISK.............." << endl;
	}
	std::vector<KeyPoint> keypoints;
	char * testtype = const_cast<char*>(testType.c_str());
	myTime *mytime = new myTime(testtype);
	mytime->start();
	ptrFeatureExact->detect(imagetest, keypoints);
	mytime->stop();
}
void RunMatchTest(string matchType, cv::Mat & src1, cv::Mat & dst1, std::vector<KeyPoint> & keypoints1, std::vector<KeyPoint> & KeyPoints2, cv::Mat & descriptors1, cv::Mat & descriptors2)
{
	FlannBasedMatcher rmatcher;
	if (matchType == string("FLANN"))
	{
		FlannBasedMatcher rmatcher;
	}
	else if (matchType == string("BRUTURE"))
	{
		BFMatcher rmatcher;
	}
	vector<DMatch> mathes;
	char * mathctype = const_cast<char*>(matchType.c_str());
	myTime * myTestTime = new myTime(mathctype);
	myTestTime->start();
	rmatcher.match(descriptors1, descriptors2, mathes);
	myTestTime->stop();
}

void RunExactDetection(cv::Mat & src1, cv::Mat & dst1, std::vector<KeyPoint> & keypoints1, std::vector<KeyPoint> & KeyPoints2,
	cv::Mat & H1to2,float & dReapeatlity,int & crossCount,cv::Ptr<FeatureDetector>  ptrExa)
{
	evaluateFeatureDetector(src1, dst1, H1to2, &keypoints1, &KeyPoints2, dReapeatlity, crossCount, ptrExa);
}


void RunRecallPrecisionCurve(std::vector<std::vector<DMatch>> & Match1to2,std::vector<std::vector<uchar>> & vecMatch,std::vector<Point2f>  & recallPrecision)
{
	computeRecallPrecisionCurve(Match1to2, vecMatch, recallPrecision);
}
class RobustMatcher {
private:
	// ���������������ָ��
	cv::Ptr<cv::FeatureDetector> detector;
	// ������������ȡ�������ָ��
	cv::Ptr<cv::DescriptorExtractor> descriptor;
	int normType;
	float ratio; // ��һ���͵ڶ��� NN ֮���������
	bool refineF; // ������� true������Ż���������
	bool refineM; // ������� true������Ż�ƥ����
	double distance; // ���������С����
	double confidence; // ���Ŷȣ����ʣ�
public:
	RobustMatcher(const cv::Ptr<cv::FeatureDetector>& detector,
		const cv::Ptr<cv::DescriptorExtractor>& descriptor =
		cv::Ptr<cv::DescriptorExtractor>()) :
		detector(detector), descriptor(descriptor),
		normType(cv::NORM_L2), ratio(0.8f),
		refineF(true), refineM(true),
		confidence(0.98), distance(1.0) {
		// ����ʹ�ù���������
		if (!this->descriptor) {
			this->descriptor = this->detector;
		}
	}
	void ConVertKeyPointToVecP(std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::Point2f> &points1)
	{
	
		for (int it = 0;it < keypoints1.size(); ++it) 
		{
			points1.push_back(keypoints1[it].pt);
		}
	}

	cv::Mat ransacTest(const std::vector<cv::DMatch>& matches,
		std::vector<cv::KeyPoint>& keypoints1,
		std::vector<cv::KeyPoint>& keypoints2,
		std::vector<cv::DMatch>& outMatches) {
		// ���ؼ���ת��Ϊ Point2f ����
		std::vector<cv::Point2f> points1, points2;
		for (std::vector<cv::DMatch>::const_iterator it = matches.begin();
			it != matches.end(); ++it) {
			// ��ȡ���ؼ����λ��
			points1.push_back(keypoints1[it->queryIdx].pt);
			// ��ȡ�Ҳ�ؼ����λ��
			points2.push_back(keypoints2[it->trainIdx].pt);
		}
		// �� RANSAC ���� F ����
		std::vector<uchar> inliers(points1.size(), 0);
		cv::Mat fundamental =
			cv::findFundamentalMat(points1,
				points2, // ƥ�����ص�
				inliers, // ƥ��״̬�� inlier �� outlier)
				cv::FM_RANSAC, // RANSAC �㷨
				distance, // ���Լ��ߵľ���
				confidence); // ���Ŷ�
				// ȡ��ʣ�µ�(inliers)ƥ����
		std::vector<uchar>::const_iterator itIn = inliers.begin();
		std::vector<cv::DMatch>::const_iterator itM = matches.begin();
		// ��������ƥ����
		for (; itIn != inliers.end(); ++itIn, ++itM) {
			if (*itIn) { // it is a valid match
				outMatches.push_back(*itM);
			}
		}
		return fundamental;
	}


	// �� RANSAC �㷨ƥ��������
// ���ػ�������������ƥ����
	cv::Mat match(cv::Mat& image1, cv::Mat& image2, // ����ͼ��
		std::vector<cv::DMatch>& matches, // ���ƥ����
		std::vector<cv::KeyPoint>& keypoints1, // ����ؼ���
		std::vector<cv::KeyPoint>& keypoints2) {
		// 1.���������
		detector->detect(image1, keypoints1);
		detector->detect(image2, keypoints2);
		// 2.��ȡ����������
		cv::Mat descriptors1, descriptors2;
		descriptor->compute(image1, keypoints1, descriptors1);
		descriptor->compute(image2, keypoints2, descriptors2);
		// 3.ƥ������ͼ��������
		// �����ڲ��ּ�ⷽ����
		// ����ƥ�����ʵ�����������飩
		cv::BFMatcher matcher(normType, // ������
			true); // �������־
			// ƥ��������
		std::vector<cv::DMatch> outputMatches;
		matcher.match(descriptors1, descriptors2, outputMatches);
		// 4.�� RANSAC �㷨��֤ƥ����
		cv::Mat fundamental = ransacTest(outputMatches,
			keypoints1, keypoints2,
			matches);
		// ���ػ�������
		return fundamental;
	}
	float checkRepateRatio(cv::Mat &Fundemental, cv::Mat &matObj, cv::Mat &matScene, std::vector<Point2f> & obj,
		std::vector<Point2f> & scene, int nGoodMatches)
	{
		int  innersize = 0;

		CvMat* pcvMat = &cvMat(Fundemental);
		const double* Hmodel = pcvMat->data.db;
		for (int isize = 0; isize < obj.size(); isize++)
		{
			double ww = 1. / (Hmodel[6] * obj[isize].x + Hmodel[7] * obj[isize].y + 1.);
			double dx = (Hmodel[0] * obj[isize].x + Hmodel[1] * obj[isize].y + Hmodel[2])*ww - scene[isize].x;
			double dy = (Hmodel[3] * obj[isize].x + Hmodel[4] * obj[isize].y + Hmodel[5])*ww - scene[isize].y;
			float err = (float)(dx*dx + dy * dy); //3������֮����Ϊ��ͬһ����
			if (err < 9)
			{
				innersize = innersize + 1;
				cout << "innersize=" << innersize << endl;
			}
		}
		//��ӡ�ڵ�ռȫ��������ı���
		float ff = (float)innersize / nGoodMatches;
		return ff;
	}
};

void RunGetMatches(std::vector<string> & strImagelist,std::vector<std::vector<DMatch>> & vecMathes,std::vector<std::vector<uchar>> & vecRightMatch)
{
	int nSize = strImagelist.size();
	int nTotal = nSize * (nSize - 1) / 2;
	vecMathes.reserve(nTotal);
	for (int i=0;i<nSize-1;++i)
	{
		for (int j=i+1;j<nSize;++j)
		{
			cv::Mat image1, image2;
			image1 = imread(strImagelist[i], IMREAD_GRAYSCALE);
			image2 = imread(strImagelist[j], IMREAD_GRAYSCALE);
			Ptr<SIFT> p1 = cv::SIFT::create(10000);
			Ptr<SIFT> p2 = cv::SIFT::create(10000);
			std::vector<KeyPoint> keypoints1, keypoints2;
			cv::Mat descriptor1, descriptor2;
			cout << "current i=" << i << "," << "j=" << j << endl;
			p1->detectAndCompute(image1, Mat(), keypoints1, descriptor1, false);
			p2->detectAndCompute(image2, Mat(), keypoints2, descriptor2, false);
			FlannBasedMatcher matcher;
			vector<DMatch> matches;
			matcher.match(descriptor1, descriptor2, matches, Mat());
			double max_dist = 0; double min_dist = 100;

			//-- Quick calculation of max and min distances between keypoints
			for (int i = 0; i < descriptor1.rows; i++)
			{
				double dist = matches[i].distance;
				if (dist < min_dist) min_dist = dist;
				if (dist > max_dist) max_dist = dist;
			}

            // printf("-- Max dist : %f \n", max_dist);
            // printf("-- Min dist : %f \n", min_dist);
			//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
			//-- PS.- radiusMatch can also be used here.
			std::vector< uchar> Right_matches;
			for (int i = 0; i < descriptor1.rows; i++)
			{
				if (matches[i].distance < 2 * min_dist)
				{
					Right_matches.push_back(1);
				}
				else
				{
					Right_matches.push_back(0);
				}
			}
			vecRightMatch.push_back(Right_matches);
			vecMathes.push_back(matches);
		}
	}

}
void FindImagelist(string &fundestring1, int nStart, int nEnd, string fundestring2, std::vector<std::string> &vecImagelist)
{
	for (int i = nStart; i <= nEnd; ++i)
	{
		string strPath = fundestring1 + std::to_string(i) + fundestring2;
		vecImagelist.push_back(strPath);
	}
}
void testHomgraphy(std::vector<Point2f>& points1, std::vector<Point2f>& points2,cv::Mat & Homgrahpy )
{
	for (int i=0;i<points1.size();++i)
	{
		cv::Mat p1,p2;
		cout << Homgrahpy.at<double>(0, 0) << endl;
		cout << Homgrahpy.at<double>(0, 1) << endl;
		cout << "test........." << endl;
		cout << Homgrahpy << endl;
		
		p1.push_back(points1[i].x);
		p1.push_back(points1[i].y);
		p1.push_back(float(1.0));
		p2.push_back(points2[i].x);
		p2.push_back(points2[i].y);
		p2.push_back(float(1.0));
		//test1
		cout << "p1:" << endl;
		cout << p1 << endl;
		cout << p1.at<float>(0, 0) << endl;
		cout << p1.at<float>(0, 1) << endl;

		cout << "p2:" << endl;
		cout << p2 << endl;

		float d1, d2, d3, d4;

		float dmu = 1.0 / (Homgrahpy.at<double>(2, 0)*p1.at<float>(0, 0) + Homgrahpy.at<double>(2, 1)*p1.at<float>(0, 1) + Homgrahpy.at<double>(2, 2));
		float dmu1 = 1.0 / (Homgrahpy.at<double>(2, 0)*p2.at<float>(0, 0) + Homgrahpy.at<double>(2, 1)*p2.at<float>(0, 1) + Homgrahpy.at<double>(2, 2));

		d1 =  (Homgrahpy.at<double>(0, 0)*p1.at<float>(0, 0) + Homgrahpy.at<double>(0, 1)*p1.at<float>(0, 1) + Homgrahpy.at<double>(0, 2))*dmu;
		d2 =  (Homgrahpy.at<double>(1, 0)*p1.at<float>(0, 0) + Homgrahpy.at<double>(1, 1)*p1.at<float>(0, 1) + Homgrahpy.at<double>(1, 2))*dmu;
		d3 =  (Homgrahpy.at<double>(0, 0)*p2.at<float>(0, 0) + Homgrahpy.at<double>(0, 1)*p2.at<float>(0, 1) + Homgrahpy.at<double>(0, 2))*dmu1;
		d4 =  (Homgrahpy.at<double>(1, 0)*p2.at<float>(0, 0) + Homgrahpy.at<double>(1, 1)*p2.at<float>(0, 1) + Homgrahpy.at<double>(1, 2))*dmu1;
		cout << "d1=" << d1 << endl;
		cout << "d2=" << d2 << endl;
		cout << "d3=" << d3 << endl;
		cout << "d4=" << d4 << endl;

	

	}

}




// ׼��ƥ��������Ĭ�ϲ�����
// SIFT �����������;
int main() {
	//��ʼ��ͼ���б�
	//Mat image1 = imread("D://data//south-building//images//P1180141.JPG", IMREAD_GRAYSCALE);
	//Mat image2 = imread("D://data//south-building//images//P1180142.JPG", IMREAD_GRAYSCALE);
	Mat image1 = imread("D:\\3DOpenSource\\graf\\img1.png", IMREAD_GRAYSCALE);
	Mat image2 = imread("D:\\3DOpenSource\\graf\\img2.png", IMREAD_GRAYSCALE);
	//�ж�ͼ���Ƿ���������
	if (!image1.data || !image2.data)
	{
		cout << "Error reading images!!" << endl;
		return -1;
	}
	//ƥ����
	std::vector<cv::DMatch> matches;
	std::vector<cv::KeyPoint> keypoints1, keypoints2, keypoints3, keypoints4, keypoints5, keypoints6, keypoints7,keypoints8;
	std::vector<cv::Point2f>  keypoints1_, keypoints2_;
	cv::Mat descriptor3, descriptor4;
	//����������ȡʱ��

	//�����Ƚ�ƥ����
	RobustMatcher rmatcher(cv::SIFT::create(1000));
	cv::Mat fundamental = rmatcher.match(image1, image2,
		matches, keypoints1, keypoints2);
	cv::Mat imageMatches;
	cv::drawMatches(image1, keypoints1,  // 1st image and its keypoints
		image2, keypoints2,  // 2nd image and its keypoints
		matches,			// the matches
		imageMatches,		// the image produced
		cv::Scalar(-1),  // color of the lines
		cv::Scalar(-1)  // color of the keypoints
	);
	//cv::imshow("Matches", imageMatches);
//cv::imwrite("D:\\data\\south-building\\images\\contactRansac.JPG", imageMatches);
	//���Բ���1
	RunExactFeatureTimeTest("SIFT", image1);
	//����sift��ȡ��
	Ptr<SIFT> pexactor1 = cv::SIFT::create(5000);
	Ptr<SIFT> pexactor2 = cv::SIFT::create(5000);
	Ptr<SIFT> pexactor3 = cv::SIFT::create();
	Ptr<SIFT> pexactor4 = cv::SIFT::create();
	//��ȡsift����
	pexactor1->detectAndCompute(image1, Mat(), keypoints3, descriptor3,false);
	pexactor2->detectAndCompute(image2, Mat(), keypoints4, descriptor4, false);
	//���Բ���2
	RunMatchTest("FLANN", image1, image2, keypoints3, keypoints4, descriptor3, descriptor4);
	//���Բ���3
	// Convert keypoints into Point2f	
	//ת������������
	std::vector<cv::Point2f> points1, points2;
	for (std::vector<cv::DMatch>::const_iterator it = matches.begin();
		it != matches.end(); ++it) {

		// Get the position of left keypoints
		float x = keypoints1[it->queryIdx].pt.x;
		float y = keypoints1[it->queryIdx].pt.y;
		points1.push_back(keypoints1[it->queryIdx].pt);
		cv::circle(image1, cv::Point(x, y), 3, cv::Scalar(255, 255, 255), 3);
		// Get the position of right keypoints
		x = keypoints2[it->trainIdx].pt.x;
		y = keypoints2[it->trainIdx].pt.y;
		cv::circle(image2, cv::Point(x, y), 3, cv::Scalar(255, 255, 255), 3);
		points2.push_back(keypoints2[it->trainIdx].pt);
	}
	BFMatcher bFmatcher;
	vector<DMatch> matches_Repeat;
	bFmatcher.match(descriptor3, descriptor4, matches_Repeat);
	std::vector<cv::Point2f> points3, points4;
	for (std::vector<cv::DMatch>::const_iterator it = matches_Repeat.begin();
		it != matches_Repeat.end(); ++it) {

		// Get the position of left keypoints
		float x = keypoints3[it->queryIdx].pt.x;
		float y = keypoints3[it->queryIdx].pt.y;
		points3.push_back(keypoints3[it->queryIdx].pt);
		cv::circle(image1, cv::Point(x, y), 3, cv::Scalar(255, 255, 255), 3);
		// Get the position of right keypoints
		x = keypoints4[it->trainIdx].pt.x;
		y = keypoints4[it->trainIdx].pt.y;
		cv::circle(image2, cv::Point(x, y), 3, cv::Scalar(255, 255, 255), 3);
		points4.push_back(keypoints2[it->trainIdx].pt);
	}
	std::vector<uchar> inliers(points4.size(), 0);
	cv::Mat  Findmeatal = cv::findFundamentalMat(points3, points4,cv::RANSAC);
	cv::Mat HomoGraphy = cv::findHomography(points3, points4, cv::RANSAC);
	cv::Mat HomoGraphy1 = cv::findHomography(points3, points4, cv::LMEDS);
	cv::Mat HomoGraphy2 = cv::findHomography(points3, points4, cv::RHO);
	cout << Findmeatal << endl;
	cout << "RANSAC:"<<HomoGraphy << endl;
	cout << "LMEDS:" << HomoGraphy1 << endl;
	cout << "RHO:" << HomoGraphy2 << endl;
	//testHomgraphy(points3, points4, HomoGraphy);
	cv::Mat h12=(cv::Mat_<double>(3,3)<<8.7976964e-01,3.1245438e-01,-3.9430589e+01,-1.8389418e-01,9.3847198e-01,1.5315784e+02,
   1.9641425e-04 ,- 1.6015275e-05  , 1.0000000e+00 );
	cout << h12 << endl;
	float reapeat = 0;
	int nCrossCount = 0;
	RunExactDetection(image1, image2, keypoints5, keypoints6,h12, reapeat, nCrossCount, pexactor3);
	cout << "reapeat=" <<reapeat<< endl;
	cout << "ncrosscount=" << nCrossCount << endl;
	//�������
	float repeat2 = 0;
	int ncurront2=0;
	RunExactDetection(image1, image2, keypoints7, keypoints8, Findmeatal, repeat2, ncurront2, pexactor4);
	cout << "reapeat2=" << repeat2 << endl;
	cout << "ncrosscount2=" << ncurront2 << endl;
	string funde1 = "D://data//south-building//images//P118014";
	string funde2 = ".JPG";
	std::vector<string> strFind;
	FindImagelist(funde1, 1, 9, funde2, strFind);
	std::vector < std::vector<DMatch> > matchesf;
	std::vector<std::vector<uchar>> vecMatch;
	RunGetMatches(strFind, matchesf,vecMatch);
	std::vector<Point2f> vecPoints;
	RunRecallPrecisionCurve(matchesf,vecMatch, vecPoints);

	// Draw the epipolar lines
	std::vector<cv::Vec3f> lines1;
	cv::computeCorrespondEpilines(points1, 1, fundamental, lines1);

	for (std::vector<cv::Vec3f>::const_iterator it = lines1.begin();
		it != lines1.end(); ++it) {

		cv::line(image2, cv::Point(0, -(*it)[2] / (*it)[1]),
			cv::Point(image2.cols, -((*it)[2] + (*it)[0] * image2.cols) / (*it)[1]),
			cv::Scalar(255, 255, 255));
	}

	std::vector<cv::Vec3f> lines2;
	cv::computeCorrespondEpilines(points2, 2, fundamental, lines2);

	for (std::vector<cv::Vec3f>::const_iterator it = lines2.begin();
		it != lines2.end(); ++it) {

		cv::line(image1, cv::Point(0, -(*it)[2] / (*it)[1]),
			cv::Point(image1.cols, -((*it)[2] + (*it)[0] * image1.cols) / (*it)[1]),
			cv::Scalar(255, 255, 255));
	}

	// Display the images with epipolar lines
	//cv::imshow("Right Image Epilines (RANSAC)", image1);
	//cv::imshow("Left Image Epilines (RANSAC)", image2);
	//cv::imwrite("D:\\data\\south-building\\images\\contactRansacEPL.JPG", image1);
	//cv::imwrite("D:\\data\\south-building\\images\\contactRansacEPR.JPG", image2);

	cv::waitKey(0);
	return 0;
}