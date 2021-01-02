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

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <fstream>
using namespace cv;
using namespace std;
struct InformationReapet
{
	int nCrosCount;
	int keypoints_left;
	int keypoints_right;
	float repeatelity;
	string groupName;
};
class myTime {

	char* name;
	double begin;
	double end;
public:
	myTime( char*pName) {
		name = new char[strlen(pName) + 1];
		std::cout << "Create clock" << pName << "starting..." << endl;
		strcpy(name, pName);							//初始化类对象名
		begin = end = 0.0;					    	//初始化数据成员begin和end

	};
	myTime(clock_t t, char* pName) {
		name = new char[strlen(pName) + 1];
		std::cout << "Create objective" << pName << "starting..." << endl;
		strcpy(name, pName);							//初始化类对象名
		begin = (double)t / CLOCKS_PER_SEC;
		end = 0.0;
	};
	~myTime() {
		std::cout << "destruct object" << name << endl;
		delete[] name;
	};

	void start() {
		begin = (double)clock() / CLOCKS_PER_SEC;
	};
	float stop() {
		end = (double)clock() / CLOCKS_PER_SEC;
		float dProcess = end - begin;
		show();
		return dProcess;
	};
	void show() {
		std::cout << "clock name is" << name <<",";
	/*	std::cout << "start：" << begin << "second" << endl;
		std::cout << "end：" << end << "second" << endl;*/
		std::cout << "timecost：" << (end - begin) << "seconds" << endl;
	};
};
void rotate_arbitrarily_angle(Mat &src, Mat &dst, float angle)
{
	float radian = (float)(angle / 180.0 * CV_PI);

	//填充图像
	int maxBorder = (int)(max(src.cols, src.rows)* 1.414); //即为sqrt(2)*max
	int dx = (maxBorder - src.cols) / 2;
	int dy = (maxBorder - src.rows) / 2;
	copyMakeBorder(src, dst, dy, dy, dx, dx, BORDER_CONSTANT);

	//旋转
	Point2f center((float)(dst.cols / 2), (float)(dst.rows / 2));
	Mat affine_matrix = getRotationMatrix2D(center, angle, 1.0);//求得旋转矩阵
	warpAffine(dst, dst, affine_matrix, dst.size());

	//计算图像旋转之后包含图像的最大的矩形
	float sinVal = abs(sin(radian));
	float cosVal = abs(cos(radian));
	Size targetSize((int)(src.cols * cosVal + src.rows * sinVal),
		(int)(src.cols * sinVal + src.rows * cosVal));

	//剪掉多余边框
	int x = (dst.cols - targetSize.width) / 2;
	int y = (dst.rows - targetSize.height) / 2;
	Rect rect(x, y, targetSize.width, targetSize.height);
	dst = Mat(dst, rect);
}
cv::Mat RansacTest(const std::vector<cv::DMatch>& matches,
	std::vector<cv::KeyPoint>& keypoints1,
	std::vector<cv::KeyPoint>& keypoints2,
	std::vector<cv::DMatch>& outMatches) {
	// 将关键点转换为 Point2f 类型
	std::vector<cv::Point2f> points1, points2;
	for (std::vector<cv::DMatch>::const_iterator it = matches.begin();
		it != matches.end(); ++it) {
		// 获取左侧关键点的位置
		points1.push_back(keypoints1[it->queryIdx].pt);
		// 获取右侧关键点的位置
		points2.push_back(keypoints2[it->trainIdx].pt);
	}
	// 用 RANSAC 计算 F 矩阵
	std::vector<uchar> inliers(points1.size(), 0);
	cv::Mat fundamental =
		cv::findFundamentalMat(points1,
			points2, // 匹配像素点
			inliers, // 匹配状态（ inlier 或 outlier)
			cv::FM_RANSAC, // RANSAC 算法
			1.0, // 到对极线的距离
			0.98); // 置信度
			// 取出剩下的(inliers)匹配项
	std::vector<uchar>::const_iterator itIn = inliers.begin();
	std::vector<cv::DMatch>::const_iterator itM = matches.begin();
	// 遍历所有匹配项
	for (; itIn != inliers.end(); ++itIn, ++itM) {
		if (*itIn) { // it is a valid match
			outMatches.push_back(*itM);
		}
	}
	return fundamental;
}
//十几件统计函数
void RunExactFeatureTimeTest(string testType,cv::Mat & imagetest)
{

	Ptr<FeatureDetector> ptrFeatureExact;
	string  strSift = "SIFT";
	if (testType==strSift)
	{
		ptrFeatureExact = cv::SIFT::create(5000, 3, 0.04, 10, 1.6);
		std::cout << "test SIFT............" << endl;
	}

	else if (testType == string("ORB"))
	{
		ptrFeatureExact = cv::ORB::create(5000);
		std::cout << "test ORB.............." << endl;
	}
	else if (testType == string("BRISK"))
	{
		ptrFeatureExact = cv::BRISK::create();
		std::cout << "test BRISK.............." << endl;
	}
	std::vector<KeyPoint> keypoints;
	cv::Mat  descriptor;
	char * testtype = const_cast<char*>(testType.c_str());
	myTime *mytime = new myTime(testtype);
	mytime->start();
	ptrFeatureExact->detectAndCompute(imagetest,Mat(), keypoints,descriptor,false);
	mytime->stop();
}

void RunDetectAndMatchTest(string testType, cv::Mat & srcmat,cv::Mat & dstmat,std::vector<KeyPoint> & key1,std::vector<KeyPoint> &key2,std::vector<DMatch>& matches
                            ,bool bUseRansac,bool bUseProvidePoints)
{
	if (!bUseProvidePoints)
	{
		Ptr<FeatureDetector> ptrFeatureExact;
		string  strSift = "SIFT";
		if (testType == strSift)
		{
			ptrFeatureExact = cv::SIFT::create(10000);
			std::cout << "test SIFT............" << endl;
		}

		else if (testType == string("ORB"))
		{
			ptrFeatureExact = cv::ORB::create(10000);
			std::cout << "test ORB.............." << endl;
		}
		else if (testType == string("BRISK"))
		{
			ptrFeatureExact = cv::BRISK::create();
			std::cout << "test BRISK.............." << endl;
		}
		if (key1.size() != 0 || key2.size() != 0 || matches.size() != 0)
		{
			key1.clear();
			key2.clear();
			matches.clear();
		}
		cv::Mat des1, des2;
		ptrFeatureExact->detectAndCompute(srcmat, Mat(), key1, des1, false);
		ptrFeatureExact->detectAndCompute(dstmat, Mat(), key2, des2, false);
		//
		BFMatcher fbmather(NORM_L2, true);
		fbmather.match(des1, des2, matches, Mat());
	}
	

	if (bUseRansac)
	{
		std::vector<DMatch> outMatches;
		RansacTest(matches, key1, key2, outMatches);
		matches.clear();
		matches = outMatches;
	}
}
void RunMatchTest(string matchType, cv::Mat & src1, cv::Mat & dst1, std::vector<KeyPoint> & keypoints1, std::vector<KeyPoint> & KeyPoints2, cv::Mat & descriptors1, cv::Mat & descriptors2)
{
    
	if (matchType == string("FLANN"))
	{
		FlannBasedMatcher rmatcher;
		vector<DMatch> mathes;
		char * mathctype = const_cast<char*>(matchType.c_str());
		myTime * myTestTime = new myTime(mathctype);
		myTestTime->start();
		rmatcher.match(descriptors1, descriptors2, mathes);
		myTestTime->stop();
	}
	else if (matchType == string("BRUTURE"))
	{
		BFMatcher rmatcher;
		vector<DMatch> mathes;
		char * mathctype = const_cast<char*>(matchType.c_str());
		myTime * myTestTime = new myTime(mathctype);
		myTestTime->start();
		rmatcher.match(descriptors1, descriptors2, mathes);
		myTestTime->stop();
	}

}

void RunExactDetection(cv::Mat & src1, cv::Mat & dst1, std::vector<KeyPoint> & keypoints1, std::vector<KeyPoint> & KeyPoints2,
	cv::Mat & H1to2,float & dReapeatlity,int & crossCount,cv::Ptr<FeatureDetector>  ptrExa)
{
	if (keypoints1.size()!=0||KeyPoints2.size()!=0)
	{
		keypoints1.clear();
		KeyPoints2.clear();
	}
	//if (ptrExa!=nullptr)
	//{
	//	ptrExa = nullptr;
	//	ptrExa = cv::SIFT::create();
	//}
	evaluateFeatureDetector(src1, dst1, H1to2, &keypoints1, &KeyPoints2, dReapeatlity, crossCount, ptrExa);
}
void DrawSaveKeypoints(cv::Mat &srImage,std::vector<KeyPoint> & vecKeypoints,std::string dstImagepath)
{
	cv::Mat dstImage;
	drawKeypoints(srImage, vecKeypoints, dstImage);
	imwrite(dstImagepath, dstImage);
}

void RunRecallPrecisionCurve(std::vector<std::vector<DMatch>> & Match1to2,std::vector<std::vector<uchar>> & vecMatch,std::vector<Point2f>  & recallPrecision)
{
	computeRecallPrecisionCurve(Match1to2, vecMatch, recallPrecision);
}
class RobustMatcher {
private:
	// 特征点检测器对象的指针
	cv::Ptr<cv::FeatureDetector> detector;
	// 特征描述子提取器对象的指针
	cv::Ptr<cv::DescriptorExtractor> descriptor;
	int normType;
	float ratio; // 第一个和第二个 NN 之间的最大比率
	bool refineF; // 如果等于 true，则会优化基础矩阵
	bool refineM; // 如果等于 true，则会优化匹配结果
	double distance; // 到极点的最小距离
	double confidence; // 可信度（概率）
public:
	RobustMatcher(const cv::Ptr<cv::FeatureDetector>& detector,
		const cv::Ptr<cv::DescriptorExtractor>& descriptor =
		cv::Ptr<cv::DescriptorExtractor>()) :
		detector(detector), descriptor(descriptor),
		normType(cv::NORM_L2), ratio(0.8f),
		refineF(true), refineM(true),
		confidence(0.98), distance(1.0) {
		// 这里使用关联描述子
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
		// 将关键点转换为 Point2f 类型
		std::vector<cv::Point2f> points1, points2;
		for (std::vector<cv::DMatch>::const_iterator it = matches.begin();
			it != matches.end(); ++it) {
			// 获取左侧关键点的位置
			points1.push_back(keypoints1[it->queryIdx].pt);
			// 获取右侧关键点的位置
			points2.push_back(keypoints2[it->trainIdx].pt);
		}
		// 用 RANSAC 计算 F 矩阵
		std::vector<uchar> inliers(points1.size(), 0);
		cv::Mat fundamental =
			cv::findFundamentalMat(points1,
				points2, // 匹配像素点
				inliers, // 匹配状态（ inlier 或 outlier)
				cv::FM_RANSAC, // RANSAC 算法
				distance, // 到对极线的距离
				confidence); // 置信度
				// 取出剩下的(inliers)匹配项
		std::vector<uchar>::const_iterator itIn = inliers.begin();
		std::vector<cv::DMatch>::const_iterator itM = matches.begin();
		// 遍历所有匹配项
		for (; itIn != inliers.end(); ++itIn, ++itM) {
			if (*itIn) { // it is a valid match
				outMatches.push_back(*itM);
			}
		}
		return fundamental;
	}


	// 用 RANSAC 算法匹配特征点
// 返回基础矩阵和输出的匹配项
	cv::Mat match(cv::Mat& image1, cv::Mat& image2, // 输入图像
		std::vector<cv::DMatch>& matches, // 输出匹配项
		std::vector<cv::KeyPoint>& keypoints1, // 输出关键点
		std::vector<cv::KeyPoint>& keypoints2) {
		// 1.检测特征点
		detector->detect(image1, keypoints1);
		detector->detect(image2, keypoints2);
		// 2.提取特征描述子
		cv::Mat descriptors1, descriptors2;
		descriptor->compute(image1, keypoints1, descriptors1);
		descriptor->compute(image2, keypoints2, descriptors2);
		// 3.匹配两幅图像描述子
		// （用于部分检测方法）
		// 构造匹配类的实例（带交叉检查）
		cv::BFMatcher matcher(normType, // 差距衡量
			true); // 交叉检查标志
			// 匹配描述子
		std::vector<cv::DMatch> outputMatches;
		matcher.match(descriptors1, descriptors2, outputMatches);
		// 4.用 RANSAC 算法验证匹配项
		cv::Mat fundamental = ransacTest(outputMatches,
			keypoints1, keypoints2,
			matches);
		// 返回基础矩阵
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
			float err = (float)(dx*dx + dy * dy); //3个像素之内认为是同一个点
			if (err < 9)
			{
				innersize = innersize + 1;
				std::cout << "innersize=" << innersize << endl;
			}
		}
		//打印内点占全部特征点的比率
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
			std::cout << "current i=" << i << "," << "j=" << j << endl;
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
		std::cout << Homgrahpy.at<double>(0, 0) << endl;
		std::cout << Homgrahpy.at<double>(0, 1) << endl;
		std::cout << "test........." << endl;
		std::cout << Homgrahpy << endl;
		
		p1.push_back(points1[i].x);
		p1.push_back(points1[i].y);
		p1.push_back(float(1.0));
		p2.push_back(points2[i].x);
		p2.push_back(points2[i].y);
		p2.push_back(float(1.0));
		//test1
		std::cout << "p1:" << endl;
		std::cout << p1 << endl;
		std::cout << p1.at<float>(0, 0) << endl;
		std::cout << p1.at<float>(0, 1) << endl;

		std::cout << "p2:" << endl;
		std::cout << p2 << endl;

		float d1, d2, d3, d4;

		float dmu = 1.0 / (Homgrahpy.at<double>(2, 0)*p1.at<float>(0, 0) + Homgrahpy.at<double>(2, 1)*p1.at<float>(0, 1) + Homgrahpy.at<double>(2, 2));
		float dmu1 = 1.0 / (Homgrahpy.at<double>(2, 0)*p2.at<float>(0, 0) + Homgrahpy.at<double>(2, 1)*p2.at<float>(0, 1) + Homgrahpy.at<double>(2, 2));

		d1 =  (Homgrahpy.at<double>(0, 0)*p1.at<float>(0, 0) + Homgrahpy.at<double>(0, 1)*p1.at<float>(0, 1) + Homgrahpy.at<double>(0, 2))*dmu;
		d2 =  (Homgrahpy.at<double>(1, 0)*p1.at<float>(0, 0) + Homgrahpy.at<double>(1, 1)*p1.at<float>(0, 1) + Homgrahpy.at<double>(1, 2))*dmu;
		d3 =  (Homgrahpy.at<double>(0, 0)*p2.at<float>(0, 0) + Homgrahpy.at<double>(0, 1)*p2.at<float>(0, 1) + Homgrahpy.at<double>(0, 2))*dmu1;
		d4 =  (Homgrahpy.at<double>(1, 0)*p2.at<float>(0, 0) + Homgrahpy.at<double>(1, 1)*p2.at<float>(0, 1) + Homgrahpy.at<double>(1, 2))*dmu1;
		std::cout << "d1=" << d1 << endl;
		std::cout << "d2=" << d2 << endl;
		std::cout << "d3=" << d3 << endl;
		std::cout << "d4=" << d4 << endl;

	}

}
void ConvertPPMToPNG(std::vector<string> & srcImageLists,std::vector<string> & dstImageLists)
{

	if (srcImageLists.size()!=dstImageLists.size())
	{
		std::cout << "Files number is not equal,please confirm" << endl;
		return;
	}
	for (int i=0;i<srcImageLists.size();++i)
	{
		cv::Mat src = imread(srcImageLists[i], IMREAD_COLOR);
		imwrite(dstImageLists[i], src);
	}
}

void ReadFeaturePoints(string &strFilepath,std::vector<KeyPoint> & vecPoints)
{
	if (vecPoints.size()!=0)
	{
		vecPoints.clear();
	}
	ifstream inFile;
	inFile.open(strFilepath);
	while (!inFile.eof())
	{
		string strLine;
		getline(inFile,strLine);
		std::vector<string> strCurrentline;
		boost::split(strCurrentline, strLine, boost::is_any_of(","), boost::token_compress_off);
		if (strCurrentline.size()==6)
		{
			KeyPoint point1_;
			point1_.pt.x = std::atof(strCurrentline[0].c_str());
			point1_.pt.y = std::atof(strCurrentline[1].c_str());
			point1_.size = std::atof(strCurrentline[2].c_str());
			vecPoints.push_back(point1_);
		}
	
	}
	inFile.close();

}
void CreatDistanceFile(std::vector<KeyPoint> & keypoints,int nWidth,string & strFile)
{

	ofstream fileout;
	fileout.open(strFile.c_str(), std::ios::out);
	int nHalfWidth = nWidth / 2;
	int nSize = keypoints.size();
	double ** dis = new double *[nSize];
	for (int i=0;i<nSize;++i)
	{
		dis[i] = new double[nSize];
		for (int j=0;j<nSize;++j)
		{
			dis[i][j] = 0;
		}
	}
	for (int i=0;i<nSize-1;++i)
	{
		for (int j=i+1;j<nSize;++j)
		{

			double dDifferx = keypoints[i].pt.x - keypoints[j].pt.x;
			double dDiffery = keypoints[i].pt.y - keypoints[j].pt.y;
			double dDistanceL2 = sqrt(dDifferx*dDifferx + dDiffery * dDiffery);
			dis[i][j] = dDistanceL2;
		}
	}
	for (int i=0;i<nSize;++i)
	{
		for (int j=0;j<i;++j)
		{
			if (i == j)
				dis[i][j] = 0;
			else
				dis[i][j] = dis[j][i];
		}
	}
	//输出到文件
	int nOverfeat = 0;
	for (int i=0;i<nSize;++i)
	{
		std::vector<double> vecMedians;
		vecMedians.reserve(nSize);
		for ( int j=0;j<nSize;++j)
		{
			fileout << dis[i][j] << ",";
			vecMedians.emplace_back(dis[i][j]);
		}
		sort(vecMedians.begin(), vecMedians.end());
		double dMeans = 0;
		if (nSize%2)
		{
			int nIndex = nSize / 2;
			dMeans = vecMedians[nIndex];
		}
		else
		{
			int nIndex = nSize / 2;
			dMeans = (vecMedians[nIndex] + vecMedians[nIndex - 1]) / 2.0;
		}
		if (dMeans>nHalfWidth)
		{
			nOverfeat++;
		}
		fileout << dMeans << std::endl;
	}
	double dOverfeat = nOverfeat * 1.0 / nSize;
	fileout << "dMean is " << dOverfeat << std::endl;
	fileout.close();

}
//检测特征点提取时间 ，匹配时间，提取点数，匹配点数，和匹配率，正确匹配率
void GetDetectAndMatch( string & strBasepath, string  strKeytype,string &strOutFile)
{

	ofstream fileout;
	fileout.open(strOutFile, std::ios::out);
	Ptr<FeatureDetector> ptrFeatureExact;
	char * typej = const_cast<char*>("SIFT");
	char * typem = const_cast<char*>("ORB");
	if (strKeytype == string("SIFT"))
	{
		ptrFeatureExact = cv::SIFT::create(6000);
		std::cout << "test SIFT............" << endl;
	}
	else if (strKeytype == string("ORB"))
	{
		ptrFeatureExact = cv::ORB::create(3000);
		std::cout << "test ORB.............." << endl;
	}
	string strdistance0= strBasepath + "distance" + strKeytype + std::to_string(0) + ".txt";
	fileout << "提取特征点的类型" << "," << "图像一特征点数量" << "," << "特征提取时间-" << "," << "图像二特征点数量" << "," << "特征提取时间二" <<
		"," << "匹配点数量" << "匹配率" << "内点率" << endl;
	string strimg1 = strBasepath + "img1.JPG";
	Mat img1 = imread(strimg1.c_str(), IMREAD_GRAYSCALE);
	myTime * pt1 = new myTime(typej);
	std::vector<KeyPoint> key1;
	cv::Mat des1;
	pt1->start();
	ptrFeatureExact->detectAndCompute(img1, Mat(), key1, des1, false);
	float dprocesstime = pt1->stop();
	fileout << key1.size() << "," << dprocesstime << ",";
	//计算距离函数
	int nMin = img1.rows < img1.cols ? img1.rows : img1.cols;
	CreatDistanceFile(key1, nMin, strdistance0);
     for (int i=0;i<5;++i)
     {
		 string strimgcurrent = strBasepath + "img" + std::to_string(i + 2) + ".JPG";
		 string strimgcontact = strBasepath + "contact" + strKeytype+std::to_string(i + 2) + ".JPG";
		 string strimggoodcontact = strBasepath + "contactgood" +strKeytype+ std::to_string(i + 2) + ".JPG";
		 string strdistance = strBasepath + "distance" + strKeytype + std::to_string(i + 2) + ".txt";
		 ofstream filediance;
		 filediance.open(strdistance.c_str(), std::ios::out);
		 Mat img2 = imread(strimgcurrent.c_str(), IMREAD_GRAYSCALE);
		 std::vector<KeyPoint>  key2;
		 cv::Mat  des2;
		 vector<DMatch> matches;
		 fileout << strKeytype << ",";
		 pt1->start();
		 ptrFeatureExact->detectAndCompute(img2, Mat(), key2, des2, false);
		 dprocesstime = pt1->stop();
		 fileout << key2.size() << "," << dprocesstime << ",";
		 //统计距离
		 int nMinWidth = img2.rows < img2.cols ? img2.rows : img2.cols;
		 CreatDistanceFile(key2, nMinWidth, strdistance);
		 BFMatcher fbmather(NORM_L2, true);
		 fbmather.match(des1, des2, matches, Mat());
		 fileout << matches.size()*2.0 / (key1.size() + key2.size())<<",";
		 std::vector<DMatch> GoodMatch;
		 RansacTest(matches, key1, key2, GoodMatch);
		 fileout << GoodMatch.size() * 1.0 / matches.size() << endl;
		 cv::Mat imageMatches, imagegoodmatch;
		 cv::drawMatches(img1, key1,  // 1st image and its keypoints
			 img2, key2,  // 2nd image and its keypoints
			 matches,			// the matches
			 imageMatches,		// the image produced
			 cv::Scalar::all(-1),  // color of the lines
			 cv::Scalar::all(-1)); // color of the keypoints
		 cv::imwrite(strimgcontact.c_str(), imageMatches);
		 cv::drawMatches(img1, key1,  // 1st image and its keypoints
			 img2, key2,  // 2nd image and its keypoints
			 GoodMatch,			// the matches
			 imagegoodmatch,		// the image produced
			 cv::Scalar::all(-1),  // color of the lines
			 cv::Scalar::all(-1)); // color of the keypoints
		 cv::imwrite(strimggoodcontact.c_str(), imagegoodmatch);

     }
	 fileout.close();



}
enum Group {
	G1,
	G2,
	G3,
	G4,
	G5,
	G6,
	G7,
	G8,
	G9,
	G10,
	G11
};
Group GetGroup(string group)
{
	if (group == string("G1"))
		return G1;
	if (group == string("G2"))
		return G2;
	if (group == string("G3"))
		return G3;
	if (group == string("G4"))
		return G4;
	if (group == string("G5"))
		return G5;
	if (group == string("G6"))
		return G6;
	if (group == string("G7"))
		return G7;
	if (group == string("G8"))
		return G8;
	if (group == string("G9"))
		return G9;
	if (group == string("G10"))
		return G10;
	if (group == string("G11"))
		return G11;

}

std::string  Getpath(string strPath)
{
	//string G1_path = "D:\\3DOpenSource\\featurepointsdata\\graf-viewpoint\\ppm\\";
	//string G2_path = "D:\\3DOpenSource\\featurepointsdata\\bark-zoomrotation\\ppm\\";
	//string G3_path = "D:\\3DOpenSource\\featurepointsdata\\bikes-blur\\ppm\\";
	//string G4_path = "D:\\3DOpenSource\\featurepointsdata\\boat-zoomrotation\\ppm\\";
	//string G5_path = "D:\\3DOpenSource\\featurepointsdata\\leuven-light\\ppm\\";
	//string G6_path = "D:\\3DOpenSource\\featurepointsdata\\trees-blur\\ppm\\";
	//string G7_path = "D:\\3DOpenSource\\featurepointsdata\\ubc-jpegcompression\\ppm\\";
	//string G8_path = "D:\\3DOpenSource\\featurepointsdata\\wall-viewpoint\\ppm\\";
	//string G9_path = "D:\\3DOpenSource\\featurepointsdata\\eb\\";
	//string G10_path = "D:\\3DOpenSource\\featurepointsdata\\uav\\";
	//string G11_path = "D:\\3DOpenSource\\featurepointsdata\\xuanzhuan\\";
	string G1_path = "G:\\data\\ursiftgroup\\g1\\";
	string G2_path = "G:\\data\\ursiftgroup\\g2\\";
	string G3_path = "G:\\data\\ursiftgroup\\g3\\";
	string G4_path = "G:\\data\\ursiftgroup\\g4\\";
	string G5_path = "G:\\data\\ursiftgroup\\g5\\";
	string G6_path = "G:\\data\\ursiftgroup\\g6\\";
	string G7_path = "G:\\data\\ursiftgroup\\g7\\";
	string G8_path = "G:\\data\\ursiftgroup\\g8\\";
	string G9_path = "G:\\data\\ursiftgroup\\g9\\";
	string G10_path = "G:\\data\\ursiftgroup\\g10\\";
	string G11_path = "G:\\data\\ursiftgroup\\g11\\";
	const Group p = GetGroup(strPath);
	switch (p)
	{
	case G1:
		return G1_path;
	case G2:
		return G2_path;
	case G3:
		return G3_path;
	case G4:
		return G4_path;
	case G5:
		return G5_path;
	case G6:
		return G6_path;
	case G7:
		return G7_path;
	case G8:
		return G8_path;
	case G9:
		return G9_path;
	case G10:
		return G10_path;
	case G11:
		return G11_path;
	default:
		break;
	}
	return "";
}

void TEST_GROUP(string  strGroup,string  strType)
{
	string strBasth = Getpath(strGroup);
	string stroutfile = strBasth +strType+"count.txt";
	GetDetectAndMatch(strBasth, strType, stroutfile);
}













// 准备匹配器（用默认参数）
// SIFT 检测器和描述;
int main() {



	
	


	//cv::Mat G1_h12=(cv::Mat_<double>(3,3)<<8.7976964e-01,3.1245438e-01,-3.9430589e+01,-1.8389418e-01,9.3847198e-01,1.5315784e+02,
 //  1.9641425e-04 ,- 1.6015275e-05  , 1.0000000e+00 );
	//cv::Mat G1_h13 = (cv::Mat_<double>(3, 3) << 7.6285898e-01, -2.9922929e-01, 2.2567123e+02
	//	, 3.3443473e-01, 1.0143901e+00, -7.6999973e+01
	//	, 3.4663091e-04, -1.4364524e-05, 1.0000000e+00);
	//cv::Mat G1_h14 = (cv::Mat_<double>(3, 3) << 6.6378505e-01  , 6.8003334e-01 ,- 3.1230335e+01
	//	,- 1.4495500e-01  , 9.7128304e-01  , 1.4877420e+02
	//	,4.2518504e-04, - 1.3930359e-05  , 1.0000000e+00);
	//cv::Mat G1_h15 = (cv::Mat_<double>(3, 3) << 6.2544644e-01  , 5.7759174e-02  , 2.2201217e+02
	//	,2.2240536e-01 ,  1.1652147e+00 ,- 2.5605611e+01
	//	,4.9212545e-04 ,- 3.6542424e-05 ,  1.0000000e+00);
	//cv::Mat G1_h16 = (cv::Mat_<double>(3, 3) << 4.2714590e-01 ,- 6.7181765e-01 , 4.5361534e+02
	//	,4.4106579e-01  , 1.0133230e+00 ,- 4.6534569e+01
	//	,5.1887712e-04 ,- 7.8853731e-05 , 1.0000000e+00);
	////Group2
	//cv::Mat G2_h12 = (cv::Mat_<double>(3, 3) << 0.7022029025774007,  0.4313737491020563 ,- 127.94661199701689
	//	,- 0.42757325092889575 , 0.6997834349758094 , 201.26193857481698
	//	,4.083733373964227E-6 , 1.5076445750988132E-5 , 1.0
	//	);
	//cv::Mat G2_h13 = (cv::Mat_<double>(3, 3) << -0.48367041358997964 ,- 0.2472935325077872,  870.2215120216712
	//	,0.29085746679198893 ,- 0.45733473891783305 , 396.1604918833091
	//	,- 3.578663704630333E-6,  6.880007548843957E-5 , 1.0);
	//cv::Mat G2_h14 = (cv::Mat_<double>(3, 3) << -0.20381418476462312 , 0.3510201271914591 , 247.1085214229702
	//	,- 0.3499531830464912 ,- 0.1975486500576974 , 466.54576370699766
	//	,- 1.5735788289619667E-5 , 1.0242951905091244E-5 ,1.0
	//	);
	//cv::Mat G2_h15 = (cv::Mat_<double>(3, 3) << 0.30558415717792214 , 0.12841186681168829 , 200.94588793078017
	//	,- 0.12861248979242065 , 0.3067557133397112  ,133.77000196887894
	//	,2.782320090398499E-6 , 5.770764104061954E-6 , 1.0);
	//cv::Mat G2_h16 = (cv::Mat_<double>(3, 3) << -0.23047631546234373 ,- 0.10655686701035443 , 583.3200507850402
	//	,0.11269946585180685 ,- 0.20718914340861153  ,355.2381263740649
	//	,- 3.580280012615393E-5 ,3.2283960511548054E-5 , 1.0);
	////Group3
	//cv::Mat G3_h12 = (cv::Mat_<double>(3, 3) << 1.0107879e+00 ,  8.2814684e-03 ,  1.8576800e+01
	//	,- 4.9128885e-03  , 1.0148779e+00 ,- 2.8851517e+01
	//	,- 1.9166087e-06 ,  8.1537620e-06  , 1.0000000e+00
	//	);
	//cv::Mat G3_h13 = (cv::Mat_<double>(3, 3) << 1.0129406e+00 ,  7.0258059e-03, - 3.5409366e+00
	//	,- 4.3550970e-03  , 1.0183920e+00, - 3.2761060e+01
	//	,- 2.9227621e-06 ,  9.0460793e-06  , 1.0000000e+00);
	//cv::Mat G3_h14 = (cv::Mat_<double>(3, 3) << 1.0201734e+00  , 1.3125949e-02 ,- 1.0048666e+01
	//	,- 7.9558939e-03 ,  1.0253060e+00 ,- 4.3000272e+01
	//	,- 2.2467584e-06 ,  1.2471581e-05  , 1.0000000e+00);
	//cv::Mat G3_h15 = (cv::Mat_<double>(3, 3) << 1.0261529e+00  , 1.2319444e-02,- 8.5197497e+00
	//	,- 8.3147838e-03 ,  1.0311644e+00 ,- 4.1319031e+01
	//	,- 9.1200792e-08  , 1.0876260e-05 ,  1.0000000e+00);
	//cv::Mat G3_h16 = (cv::Mat_<double>(3, 3) << 1.0427236e+00  , 1.2359858e-02 ,- 1.6974167e+01
	//	,- 4.2238744e-03  , 1.0353397e+00 ,- 4.5312478e+01
	//	,1.2020516e-05 ,  8.2950327e-06  , 1.0000000e+00);
	////Group4
	//cv::Mat G4_h12 = (cv::Mat_<double>(3, 3) << 8.5828552e-01 ,  2.1564369e-01  , 9.9101418e+00
	//	,- 2.1158440e-01 ,  8.5876360e-01  , 1.3047838e+02
	//	,2.0702435e-06 ,  1.2886110e-06  , 1.0000000e+00);
	//cv::Mat G4_h13 = (cv::Mat_<double>(3, 3) << 5.6887079e-01 ,  4.6997572e-01 ,  2.5515642e+01
	//	,- 4.6783159e-01  , 5.6548769e-01  , 3.4819925e+02
	//	,6.4697420e-06, - 1.1704138e-06  , 1.0000000e+00);
	//cv::Mat G4_h14 = (cv::Mat_<double>(3, 3) << 1.0016637e-01 ,  5.2319717e-01  , 2.0587932e+02
	//	,- 5.2345249e-01 , 8.7390786e-02  , 5.3454522e+02
	//	,9.4931475e-06 ,- 9.8296917e-06 ,  1.0000000e+00);
	//cv::Mat G4_h15 = (cv::Mat_<double>(3, 3) << 4.2310823e-01 ,- 6.0670438e-02  , 2.6635003e+02
	//	,6.2730152e-02  , 4.1652096e-01  , 1.7460201e+02
	//	,1.5812849e-05 ,- 1.4368783e-05  , 1.0000000e+00);
	//cv::Mat G4_h16 = (cv::Mat_<double>(3, 3) << 2.9992872e-01  , 2.2821975e-01  , 2.2930182e+02
	//	,- 2.3832758e-01  , 2.4564042e-01 ,  3.6767399e+02
	//	,9.9064973e-05 ,- 5.8498673e-05 ,  1.0000000e+00);
	////Group5
	//cv::Mat G5_h12 = (cv::Mat_<double>(3, 3) << 5.7783232e-01 ,- 1.8122966e-04 ,  2.8225664e+00
	//	,2.2114401e-03 ,  5.7937539e-01 ,- 1.7879175e+00
	//	,- 2.3911512e-06 ,  2.9032886e-06 ,  5.7865196e-01);
	//cv::Mat G5_h13 = (cv::Mat_<double>(3, 3) << 5.7386650e-01  , 3.2769965e-03  , 2.8696063e+00
	//	,- 4.2640197e-04 ,  5.7654920e-01 ,- 2.6477989e+00
	//	,- 3.5986087e-06 ,  5.4939501e-06 ,  5.7489565e-01);
	//cv::Mat G5_h14 = (cv::Mat_<double>(3, 3) << 5.7494804e-01 ,  2.7800742e-03  , 4.9723266e+00
	//	,1.7588927e-03  , 5.7873002e-01 ,- 5.4767862e+00
	//	,- 4.9951367e-06 ,  8.0784390e-06 ,  5.7639952e-01);
	//cv::Mat G5_h15 = (cv::Mat_<double>(3, 3) << -5.7780461e-01 ,- 7.4653534e-03, - 1.8849466e-01
	//	,8.4414658e-04 ,- 5.8051698e-01  , 4.5205045e+00
	//	,2.0525034e-06 ,- 1.0250081e-05 ,- 5.7604815e-01);
	//cv::Mat G5_h16 = (cv::Mat_<double>(3, 3) << 5.8695833e-01 ,  6.2763397e-03  , 1.3078972e+00
	//	,1.9788878e-03  , 5.8978058e-01 ,- 9.5598967e+00
	//	,- 1.6508045e-06  , 1.3162429e-05  , 5.8394502e-01);
	////Group6
	//cv::Mat G6_h12 = (cv::Mat_<double>(3, 3) << 0.9912089374539974,  0.04561277689934186 , 16.430575146746467
	//	,- 0.047962294548037 , 0.9957951079956443  ,17.73539383094122
	//	,- 8.73992330142086E-6 , 1.1499680166976166E-6 , 1.0);
	//cv::Mat G6_h13 = (cv::Mat_<double>(3, 3) << 0.9905632952782818 , 0.043347675611079745  ,8.236464788207753
	//	,- 0.04702626594025669 , 0.9988185365751873 , 17.373870105550285
	//	,- 1.5932518558327435E-5 , 2.893846251213057E-6 , 1.0);
	//cv::Mat G6_h14 = (cv::Mat_<double>(3, 3) << 1.0263805284648657 , 0.04713298536155905 ,- 16.259771505387544
	//	,- 0.038844062111074 , 1.0188531347224243  ,0.6449843282481993
	//	,7.567223321612053E-6 , 3.665043946826549E-6 , 1.0);
	//cv::Mat G6_h15 = (cv::Mat_<double>(3, 3) << 1.0222521389207018 , 0.04749404190465927 ,- 25.968189130651552
	//	,- 0.04681492541525158 , 1.0157636596278663  ,10.560803317308023
	//	,3.244887964210479E-6 ,- 3.4221108923317904E-6 , 1.0);
	//cv::Mat G6_h16 = (cv::Mat_<double>(3, 3) << 1.0364619265098058 , 0.054448231785273325 ,- 26.573496349036247
	//	,- 0.050205448575418116 , 1.022285037696358 , 9.089883116763504
	//	,1.0110609732276445E-5 ,- 6.405721835180334E-6 , 1.0);
	////Group7
	//cv::Mat G7_h12 = (cv::Mat_<double>(3, 3) << 1, 0 ,0
	//	,0 ,1 ,0
	//	,0, 0 ,1);
	//cv::Mat G7_h13 = (cv::Mat_<double>(3, 3) << 1, 0 ,0
	//	,0 ,1 ,0
	//	,0, 0 ,1);
	//cv::Mat G7_h14 = (cv::Mat_<double>(3, 3) << 1, 0 ,0
	//	,0 ,1 ,0
	//	,0, 0, 1);
	//cv::Mat G7_h15 = (cv::Mat_<double>(3, 3) << 1 ,0 ,0
	//	,0, 1, 0
	//	,0 ,0, 1);
	//cv::Mat G7_h16 = (cv::Mat_<double>(3, 3) << 1 ,0 ,0
	//	,0 ,1 ,0
	//	,0 ,0, 1);
	////Group8
	//cv::Mat G8_h12 = (cv::Mat_<double>(3, 3) << 0.7882767153207999 , 0.010905680735846527 , 28.170495497465602
	//	,- 0.02537010994777608 , 0.9232684706505401 , 44.20085016989556
	//	,- 1.1457814415224265E-4  ,1.288160474307972E-5 , 1.0);
	//cv::Mat G8_h13 = (cv::Mat_<double>(3, 3) << 0.6682947339156113,  0.018344318347851395,  39.51916188173466
	//	,- 0.04902747132995888 , 0.8935012492790394 , 61.81007229702091
	//	,- 1.8999645773011534E-4 , 2.069199620253009E-6 , 1.0);
	//cv::Mat G8_h14 = (cv::Mat_<double>(3, 3) << 0.5487967233294201 , 0.015245351406439072 , 65.03321744618472
	//	,- 0.06274161341697451 , 0.8804280211603792 , 105.39150873162244
	//	,- 2.469232356641658E-4 , 1.6209582458142305E-5  ,1.0);
	//cv::Mat G8_h15 = (cv::Mat_<double>(3, 3) << 0.4133591554597665 , 0.026091530324690363 , 61.728731455568294
	//	,- 0.08735789108803846 , 0.8660455372469953 , 95.35388347437842
	//	,- 3.3657799873241785E-4 ,- 8.590195344076651E-6 , 1.0);
	//cv::Mat G8_h16 = (cv::Mat_<double>(3, 3) << 0.25816310023931976 , 0.028122203548214684  ,122.77193310808889
	//	,- 0.09827340705979042 , 0.9034280861806072  ,87.97366097395911
	//	,- 4.096870682438919E-4 , 8.67484264796887E-7 , 1.0);
	
//test Group
	{
	string strType = "SIFT";
	TEST_GROUP("G1", strType);
	TEST_GROUP("G2", strType);
	TEST_GROUP("G3", strType);
	TEST_GROUP("G4", strType);
	TEST_GROUP("G5", strType);
	TEST_GROUP("G6", strType);
	//TEST_GROUP("G7", strType);
}

















		
	


}