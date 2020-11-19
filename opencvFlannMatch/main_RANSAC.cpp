//#include <stdio.h>
//#include <time.h>
//#include <iostream>
//#include <opencv2/core/utility.hpp>
//#include "opencv2/core/core.hpp"
//#include <opencv2/core/types_c.h>
//#include "opencv2/features2d/features2d.hpp"
//
//#include "opencv2/highgui/highgui.hpp"
//#include <opencv2/calib3d.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/features2d.hpp>
//
//#include <boost/algorithm/string/classification.hpp>
//#include <boost/algorithm/string/split.hpp>
//#include <fstream>
//using namespace cv;
//using namespace std;
//struct InformationReapet
//{
//	int nCrosCount;
//	int keypoints_left;
//	int keypoints_right;
//	float repeatelity;
//	string groupName;
//};
//class myTime {
//
//	char* name;
//	double begin;
//	double end;
//public:
//	myTime( char*pName) {
//		name = new char[strlen(pName) + 1];
//		std::cout << "Create clock" << pName << "starting..." << endl;
//		strcpy(name, pName);							//初始化类对象名
//		begin = end = 0.0;					    	//初始化数据成员begin和end
//
//	};
//	myTime(clock_t t, char* pName) {
//		name = new char[strlen(pName) + 1];
//		std::cout << "Create objective" << pName << "starting..." << endl;
//		strcpy(name, pName);							//初始化类对象名
//		begin = (double)t / CLOCKS_PER_SEC;
//		end = 0.0;
//	};
//	~myTime() {
//		std::cout << "destruct object" << name << endl;
//		delete[] name;
//	};
//
//	void start() {
//		begin = (double)clock() / CLOCKS_PER_SEC;
//	};
//	void stop() {
//		end = (double)clock() / CLOCKS_PER_SEC;
//		show();
//	};
//	void show() {
//		std::cout << "clock name is" << name <<",";
//	/*	std::cout << "start：" << begin << "second" << endl;
//		std::cout << "end：" << end << "second" << endl;*/
//		std::cout << "timecost：" << (end - begin) << "seconds" << endl;
//	};
//};
//void rotate_arbitrarily_angle(Mat &src, Mat &dst, float angle)
//{
//	float radian = (float)(angle / 180.0 * CV_PI);
//
//	//填充图像
//	int maxBorder = (int)(max(src.cols, src.rows)* 1.414); //即为sqrt(2)*max
//	int dx = (maxBorder - src.cols) / 2;
//	int dy = (maxBorder - src.rows) / 2;
//	copyMakeBorder(src, dst, dy, dy, dx, dx, BORDER_CONSTANT);
//
//	//旋转
//	Point2f center((float)(dst.cols / 2), (float)(dst.rows / 2));
//	Mat affine_matrix = getRotationMatrix2D(center, angle, 1.0);//求得旋转矩阵
//	warpAffine(dst, dst, affine_matrix, dst.size());
//
//	//计算图像旋转之后包含图像的最大的矩形
//	float sinVal = abs(sin(radian));
//	float cosVal = abs(cos(radian));
//	Size targetSize((int)(src.cols * cosVal + src.rows * sinVal),
//		(int)(src.cols * sinVal + src.rows * cosVal));
//
//	//剪掉多余边框
//	int x = (dst.cols - targetSize.width) / 2;
//	int y = (dst.rows - targetSize.height) / 2;
//	Rect rect(x, y, targetSize.width, targetSize.height);
//	dst = Mat(dst, rect);
//}
//cv::Mat RansacTest(const std::vector<cv::DMatch>& matches,
//	std::vector<cv::KeyPoint>& keypoints1,
//	std::vector<cv::KeyPoint>& keypoints2,
//	std::vector<cv::DMatch>& outMatches) {
//	// 将关键点转换为 Point2f 类型
//	std::vector<cv::Point2f> points1, points2;
//	for (std::vector<cv::DMatch>::const_iterator it = matches.begin();
//		it != matches.end(); ++it) {
//		// 获取左侧关键点的位置
//		points1.push_back(keypoints1[it->queryIdx].pt);
//		// 获取右侧关键点的位置
//		points2.push_back(keypoints2[it->trainIdx].pt);
//	}
//	// 用 RANSAC 计算 F 矩阵
//	std::vector<uchar> inliers(points1.size(), 0);
//	cv::Mat fundamental =
//		cv::findFundamentalMat(points1,
//			points2, // 匹配像素点
//			inliers, // 匹配状态（ inlier 或 outlier)
//			cv::FM_RANSAC, // RANSAC 算法
//			1.0, // 到对极线的距离
//			0.98); // 置信度
//			// 取出剩下的(inliers)匹配项
//	std::vector<uchar>::const_iterator itIn = inliers.begin();
//	std::vector<cv::DMatch>::const_iterator itM = matches.begin();
//	// 遍历所有匹配项
//	for (; itIn != inliers.end(); ++itIn, ++itM) {
//		if (*itIn) { // it is a valid match
//			outMatches.push_back(*itM);
//		}
//	}
//	return fundamental;
//}
////十几件统计函数
//void RunExactFeatureTimeTest(string testType,cv::Mat & imagetest)
//{
//
//	Ptr<FeatureDetector> ptrFeatureExact;
//	string  strSift = "SIFT";
//	if (testType==strSift)
//	{
//		ptrFeatureExact = cv::SIFT::create(5000, 3, 0.04, 10, 1.6);
//		std::cout << "test SIFT............" << endl;
//	}
//
//	else if (testType == string("ORB"))
//	{
//		ptrFeatureExact = cv::ORB::create(5000);
//		std::cout << "test ORB.............." << endl;
//	}
//	else if (testType == string("BRISK"))
//	{
//		ptrFeatureExact = cv::BRISK::create();
//		std::cout << "test BRISK.............." << endl;
//	}
//	std::vector<KeyPoint> keypoints;
//	cv::Mat  descriptor;
//	char * testtype = const_cast<char*>(testType.c_str());
//	myTime *mytime = new myTime(testtype);
//	mytime->start();
//	ptrFeatureExact->detectAndCompute(imagetest,Mat(), keypoints,descriptor,false);
//	mytime->stop();
//}
//
//void RunDetectAndMatchTest(string testType, cv::Mat & srcmat,cv::Mat & dstmat,std::vector<KeyPoint> & key1,std::vector<KeyPoint> &key2,std::vector<DMatch>& matches
//                            ,bool bUseRansac,bool bUseProvidePoints)
//{
//	if (!bUseProvidePoints)
//	{
//		Ptr<FeatureDetector> ptrFeatureExact;
//		string  strSift = "SIFT";
//		if (testType == strSift)
//		{
//			ptrFeatureExact = cv::SIFT::create(10000);
//			std::cout << "test SIFT............" << endl;
//		}
//
//		else if (testType == string("ORB"))
//		{
//			ptrFeatureExact = cv::ORB::create(10000);
//			std::cout << "test ORB.............." << endl;
//		}
//		else if (testType == string("BRISK"))
//		{
//			ptrFeatureExact = cv::BRISK::create();
//			std::cout << "test BRISK.............." << endl;
//		}
//		if (key1.size() != 0 || key2.size() != 0 || matches.size() != 0)
//		{
//			key1.clear();
//			key2.clear();
//			matches.clear();
//		}
//		cv::Mat des1, des2;
//		ptrFeatureExact->detectAndCompute(srcmat, Mat(), key1, des1, false);
//		ptrFeatureExact->detectAndCompute(dstmat, Mat(), key2, des2, false);
//		//
//		BFMatcher fbmather(NORM_L2, true);
//		fbmather.match(des1, des2, matches, Mat());
//	}
//	
//
//	if (bUseRansac)
//	{
//		std::vector<DMatch> outMatches;
//		RansacTest(matches, key1, key2, outMatches);
//		matches.clear();
//		matches = outMatches;
//	}
//}
//void RunMatchTest(string matchType, cv::Mat & src1, cv::Mat & dst1, std::vector<KeyPoint> & keypoints1, std::vector<KeyPoint> & KeyPoints2, cv::Mat & descriptors1, cv::Mat & descriptors2)
//{
//    
//	if (matchType == string("FLANN"))
//	{
//		FlannBasedMatcher rmatcher;
//		vector<DMatch> mathes;
//		char * mathctype = const_cast<char*>(matchType.c_str());
//		myTime * myTestTime = new myTime(mathctype);
//		myTestTime->start();
//		rmatcher.match(descriptors1, descriptors2, mathes);
//		myTestTime->stop();
//	}
//	else if (matchType == string("BRUTURE"))
//	{
//		BFMatcher rmatcher;
//		vector<DMatch> mathes;
//		char * mathctype = const_cast<char*>(matchType.c_str());
//		myTime * myTestTime = new myTime(mathctype);
//		myTestTime->start();
//		rmatcher.match(descriptors1, descriptors2, mathes);
//		myTestTime->stop();
//	}
//
//}
//
//void RunExactDetection(cv::Mat & src1, cv::Mat & dst1, std::vector<KeyPoint> & keypoints1, std::vector<KeyPoint> & KeyPoints2,
//	cv::Mat & H1to2,float & dReapeatlity,int & crossCount,cv::Ptr<FeatureDetector>  ptrExa)
//{
//	if (keypoints1.size()!=0||KeyPoints2.size()!=0)
//	{
//		keypoints1.clear();
//		KeyPoints2.clear();
//	}
//	//if (ptrExa!=nullptr)
//	//{
//	//	ptrExa = nullptr;
//	//	ptrExa = cv::SIFT::create();
//	//}
//	evaluateFeatureDetector(src1, dst1, H1to2, &keypoints1, &KeyPoints2, dReapeatlity, crossCount, ptrExa);
//}
//void DrawSaveKeypoints(cv::Mat &srImage,std::vector<KeyPoint> & vecKeypoints,std::string dstImagepath)
//{
//	cv::Mat dstImage;
//	drawKeypoints(srImage, vecKeypoints, dstImage);
//	imwrite(dstImagepath, dstImage);
//}
//
//void RunRecallPrecisionCurve(std::vector<std::vector<DMatch>> & Match1to2,std::vector<std::vector<uchar>> & vecMatch,std::vector<Point2f>  & recallPrecision)
//{
//	computeRecallPrecisionCurve(Match1to2, vecMatch, recallPrecision);
//}
//class RobustMatcher {
//private:
//	// 特征点检测器对象的指针
//	cv::Ptr<cv::FeatureDetector> detector;
//	// 特征描述子提取器对象的指针
//	cv::Ptr<cv::DescriptorExtractor> descriptor;
//	int normType;
//	float ratio; // 第一个和第二个 NN 之间的最大比率
//	bool refineF; // 如果等于 true，则会优化基础矩阵
//	bool refineM; // 如果等于 true，则会优化匹配结果
//	double distance; // 到极点的最小距离
//	double confidence; // 可信度（概率）
//public:
//	RobustMatcher(const cv::Ptr<cv::FeatureDetector>& detector,
//		const cv::Ptr<cv::DescriptorExtractor>& descriptor =
//		cv::Ptr<cv::DescriptorExtractor>()) :
//		detector(detector), descriptor(descriptor),
//		normType(cv::NORM_L2), ratio(0.8f),
//		refineF(true), refineM(true),
//		confidence(0.98), distance(1.0) {
//		// 这里使用关联描述子
//		if (!this->descriptor) {
//			this->descriptor = this->detector;
//		}
//	}
//	void ConVertKeyPointToVecP(std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::Point2f> &points1)
//	{
//	
//		for (int it = 0;it < keypoints1.size(); ++it) 
//		{
//			points1.push_back(keypoints1[it].pt);
//		}
//	}
//
//	cv::Mat ransacTest(const std::vector<cv::DMatch>& matches,
//		std::vector<cv::KeyPoint>& keypoints1,
//		std::vector<cv::KeyPoint>& keypoints2,
//		std::vector<cv::DMatch>& outMatches) {
//		// 将关键点转换为 Point2f 类型
//		std::vector<cv::Point2f> points1, points2;
//		for (std::vector<cv::DMatch>::const_iterator it = matches.begin();
//			it != matches.end(); ++it) {
//			// 获取左侧关键点的位置
//			points1.push_back(keypoints1[it->queryIdx].pt);
//			// 获取右侧关键点的位置
//			points2.push_back(keypoints2[it->trainIdx].pt);
//		}
//		// 用 RANSAC 计算 F 矩阵
//		std::vector<uchar> inliers(points1.size(), 0);
//		cv::Mat fundamental =
//			cv::findFundamentalMat(points1,
//				points2, // 匹配像素点
//				inliers, // 匹配状态（ inlier 或 outlier)
//				cv::FM_RANSAC, // RANSAC 算法
//				distance, // 到对极线的距离
//				confidence); // 置信度
//				// 取出剩下的(inliers)匹配项
//		std::vector<uchar>::const_iterator itIn = inliers.begin();
//		std::vector<cv::DMatch>::const_iterator itM = matches.begin();
//		// 遍历所有匹配项
//		for (; itIn != inliers.end(); ++itIn, ++itM) {
//			if (*itIn) { // it is a valid match
//				outMatches.push_back(*itM);
//			}
//		}
//		return fundamental;
//	}
//
//
//	// 用 RANSAC 算法匹配特征点
//// 返回基础矩阵和输出的匹配项
//	cv::Mat match(cv::Mat& image1, cv::Mat& image2, // 输入图像
//		std::vector<cv::DMatch>& matches, // 输出匹配项
//		std::vector<cv::KeyPoint>& keypoints1, // 输出关键点
//		std::vector<cv::KeyPoint>& keypoints2) {
//		// 1.检测特征点
//		detector->detect(image1, keypoints1);
//		detector->detect(image2, keypoints2);
//		// 2.提取特征描述子
//		cv::Mat descriptors1, descriptors2;
//		descriptor->compute(image1, keypoints1, descriptors1);
//		descriptor->compute(image2, keypoints2, descriptors2);
//		// 3.匹配两幅图像描述子
//		// （用于部分检测方法）
//		// 构造匹配类的实例（带交叉检查）
//		cv::BFMatcher matcher(normType, // 差距衡量
//			true); // 交叉检查标志
//			// 匹配描述子
//		std::vector<cv::DMatch> outputMatches;
//		matcher.match(descriptors1, descriptors2, outputMatches);
//		// 4.用 RANSAC 算法验证匹配项
//		cv::Mat fundamental = ransacTest(outputMatches,
//			keypoints1, keypoints2,
//			matches);
//		// 返回基础矩阵
//		return fundamental;
//	}
//	float checkRepateRatio(cv::Mat &Fundemental, cv::Mat &matObj, cv::Mat &matScene, std::vector<Point2f> & obj,
//		std::vector<Point2f> & scene, int nGoodMatches)
//	{
//		int  innersize = 0;
//
//		CvMat* pcvMat = &cvMat(Fundemental);
//		const double* Hmodel = pcvMat->data.db;
//		for (int isize = 0; isize < obj.size(); isize++)
//		{
//			double ww = 1. / (Hmodel[6] * obj[isize].x + Hmodel[7] * obj[isize].y + 1.);
//			double dx = (Hmodel[0] * obj[isize].x + Hmodel[1] * obj[isize].y + Hmodel[2])*ww - scene[isize].x;
//			double dy = (Hmodel[3] * obj[isize].x + Hmodel[4] * obj[isize].y + Hmodel[5])*ww - scene[isize].y;
//			float err = (float)(dx*dx + dy * dy); //3个像素之内认为是同一个点
//			if (err < 9)
//			{
//				innersize = innersize + 1;
//				std::cout << "innersize=" << innersize << endl;
//			}
//		}
//		//打印内点占全部特征点的比率
//		float ff = (float)innersize / nGoodMatches;
//		return ff;
//	}
//};
//
//void RunGetMatches(std::vector<string> & strImagelist,std::vector<std::vector<DMatch>> & vecMathes,std::vector<std::vector<uchar>> & vecRightMatch)
//{
//	int nSize = strImagelist.size();
//	int nTotal = nSize * (nSize - 1) / 2;
//	vecMathes.reserve(nTotal);
//	for (int i=0;i<nSize-1;++i)
//	{
//		for (int j=i+1;j<nSize;++j)
//		{
//			cv::Mat image1, image2;
//			image1 = imread(strImagelist[i], IMREAD_GRAYSCALE);
//			image2 = imread(strImagelist[j], IMREAD_GRAYSCALE);
//			Ptr<SIFT> p1 = cv::SIFT::create(10000);
//			Ptr<SIFT> p2 = cv::SIFT::create(10000);
//			std::vector<KeyPoint> keypoints1, keypoints2;
//			cv::Mat descriptor1, descriptor2;
//			std::cout << "current i=" << i << "," << "j=" << j << endl;
//			p1->detectAndCompute(image1, Mat(), keypoints1, descriptor1, false);
//			p2->detectAndCompute(image2, Mat(), keypoints2, descriptor2, false);
//			FlannBasedMatcher matcher;
//			vector<DMatch> matches;
//			matcher.match(descriptor1, descriptor2, matches, Mat());
//			double max_dist = 0; double min_dist = 100;
//
//			//-- Quick calculation of max and min distances between keypoints
//			for (int i = 0; i < descriptor1.rows; i++)
//			{
//				double dist = matches[i].distance;
//				if (dist < min_dist) min_dist = dist;
//				if (dist > max_dist) max_dist = dist;
//			}
//
//            // printf("-- Max dist : %f \n", max_dist);
//            // printf("-- Min dist : %f \n", min_dist);
//			//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
//			//-- PS.- radiusMatch can also be used here.
//			std::vector< uchar> Right_matches;
//			for (int i = 0; i < descriptor1.rows; i++)
//			{
//				if (matches[i].distance < 2 * min_dist)
//				{
//					Right_matches.push_back(1);
//				}
//				else
//				{
//					Right_matches.push_back(0);
//				}
//			}
//			vecRightMatch.push_back(Right_matches);
//			vecMathes.push_back(matches);
//		}
//	}
//
//}
//void FindImagelist(string &fundestring1, int nStart, int nEnd, string fundestring2, std::vector<std::string> &vecImagelist)
//{
//	for (int i = nStart; i <= nEnd; ++i)
//	{
//		string strPath = fundestring1 + std::to_string(i) + fundestring2;
//		vecImagelist.push_back(strPath);
//	}
//}
//void testHomgraphy(std::vector<Point2f>& points1, std::vector<Point2f>& points2,cv::Mat & Homgrahpy )
//{
//	for (int i=0;i<points1.size();++i)
//	{
//		cv::Mat p1,p2;
//		std::cout << Homgrahpy.at<double>(0, 0) << endl;
//		std::cout << Homgrahpy.at<double>(0, 1) << endl;
//		std::cout << "test........." << endl;
//		std::cout << Homgrahpy << endl;
//		
//		p1.push_back(points1[i].x);
//		p1.push_back(points1[i].y);
//		p1.push_back(float(1.0));
//		p2.push_back(points2[i].x);
//		p2.push_back(points2[i].y);
//		p2.push_back(float(1.0));
//		//test1
//		std::cout << "p1:" << endl;
//		std::cout << p1 << endl;
//		std::cout << p1.at<float>(0, 0) << endl;
//		std::cout << p1.at<float>(0, 1) << endl;
//
//		std::cout << "p2:" << endl;
//		std::cout << p2 << endl;
//
//		float d1, d2, d3, d4;
//
//		float dmu = 1.0 / (Homgrahpy.at<double>(2, 0)*p1.at<float>(0, 0) + Homgrahpy.at<double>(2, 1)*p1.at<float>(0, 1) + Homgrahpy.at<double>(2, 2));
//		float dmu1 = 1.0 / (Homgrahpy.at<double>(2, 0)*p2.at<float>(0, 0) + Homgrahpy.at<double>(2, 1)*p2.at<float>(0, 1) + Homgrahpy.at<double>(2, 2));
//
//		d1 =  (Homgrahpy.at<double>(0, 0)*p1.at<float>(0, 0) + Homgrahpy.at<double>(0, 1)*p1.at<float>(0, 1) + Homgrahpy.at<double>(0, 2))*dmu;
//		d2 =  (Homgrahpy.at<double>(1, 0)*p1.at<float>(0, 0) + Homgrahpy.at<double>(1, 1)*p1.at<float>(0, 1) + Homgrahpy.at<double>(1, 2))*dmu;
//		d3 =  (Homgrahpy.at<double>(0, 0)*p2.at<float>(0, 0) + Homgrahpy.at<double>(0, 1)*p2.at<float>(0, 1) + Homgrahpy.at<double>(0, 2))*dmu1;
//		d4 =  (Homgrahpy.at<double>(1, 0)*p2.at<float>(0, 0) + Homgrahpy.at<double>(1, 1)*p2.at<float>(0, 1) + Homgrahpy.at<double>(1, 2))*dmu1;
//		std::cout << "d1=" << d1 << endl;
//		std::cout << "d2=" << d2 << endl;
//		std::cout << "d3=" << d3 << endl;
//		std::cout << "d4=" << d4 << endl;
//
//	}
//
//}
//void ConvertPPMToPNG(std::vector<string> & srcImageLists,std::vector<string> & dstImageLists)
//{
//
//	if (srcImageLists.size()!=dstImageLists.size())
//	{
//		std::cout << "Files number is not equal,please confirm" << endl;
//		return;
//	}
//	for (int i=0;i<srcImageLists.size();++i)
//	{
//		cv::Mat src = imread(srcImageLists[i], IMREAD_COLOR);
//		imwrite(dstImageLists[i], src);
//	}
//}
//
//void ReadFeaturePoints(string &strFilepath,std::vector<KeyPoint> & vecPoints)
//{
//	if (vecPoints.size()!=0)
//	{
//		vecPoints.clear();
//	}
//	ifstream inFile;
//	inFile.open(strFilepath);
//	while (!inFile.eof())
//	{
//		string strLine;
//		getline(inFile,strLine);
//		std::vector<string> strCurrentline;
//		boost::split(strCurrentline, strLine, boost::is_any_of(","), boost::token_compress_off);
//		if (strCurrentline.size()==6)
//		{
//			KeyPoint point1_;
//			point1_.pt.x = std::atof(strCurrentline[0].c_str());
//			point1_.pt.y = std::atof(strCurrentline[1].c_str());
//			point1_.size = std::atof(strCurrentline[2].c_str());
//			vecPoints.push_back(point1_);
//		}
//	
//	}
//	inFile.close();
//
//}
//
//// 准备匹配器（用默认参数）
//// SIFT 检测器和描述;
//int main() {
//	//初始化图像列表
//	//Mat image1 = imread("D://data//south-building//images//P1180141.JPG", IMREAD_GRAYSCALE);
//	//Mat image2 = imread("D://data//south-building//images//P1180142.JPG", IMREAD_GRAYSCALE);
//	Mat image1 = imread("D:\\3DOpenSource\\featurepointsdata\\graf-viewpoint\\ppm\\img1.png", IMREAD_GRAYSCALE);
//	Mat image2 = imread("D:\\3DOpenSource\\featurepointsdata\\graf-viewpoint\\ppm\\img2.png", IMREAD_GRAYSCALE);
//	//重复率数字
//	string resultPath = "D:\\3DOpenSource\\featurepointsdata\\repeat_sift5000.txt";
//	string resultURsiftPath = "D:\\3DOpenSource\\featurepointsdata\\repeat_ursift5000_50.txt";
//	//Group1 graf-viewpoint
//		Mat G1_image1 = imread("D:\\3DOpenSource\\featurepointsdata\\graf-viewpoint\\ppm\\img1.png", IMREAD_GRAYSCALE);
//		//Mat G1_image2 = imread("D:\\3DOpenSource\\featurepointsdata\\graf-viewpoint\\ppm\\img2.png", IMREAD_GRAYSCALE);
//		Mat G1_image3 = imread("D:\\3DOpenSource\\featurepointsdata\\graf-viewpoint\\ppm\\img3.png", IMREAD_GRAYSCALE);
//		Mat G1_image4 = imread("D:\\3DOpenSource\\featurepointsdata\\graf-viewpoint\\ppm\\img4.png", IMREAD_GRAYSCALE);
//		Mat G1_image5 = imread("D:\\3DOpenSource\\featurepointsdata\\graf-viewpoint\\ppm\\img5.png", IMREAD_GRAYSCALE);
//		Mat G1_image6 = imread("D:\\3DOpenSource\\featurepointsdata\\graf-viewpoint\\ppm\\img6.png", IMREAD_GRAYSCALE);
//		Mat G1_image2 = imread("D:\\3DOpenSource\\featurepointsdata\\graf-viewpoint\\ppm\\test1.JPG", IMREAD_GRAYSCALE);
//		string G1_path = "D:\\3DOpenSource\\featurepointsdata\\graf-viewpoint\\ppm\\";
//	//Group2 bark-zoomrotation
//		Mat G2_image1 = imread("D:\\3DOpenSource\\featurepointsdata\\bark-zoomrotation\\ppm\\img1.png", IMREAD_GRAYSCALE);
//		Mat G2_image2 = imread("D:\\3DOpenSource\\featurepointsdata\\bark-zoomrotation\\ppm\\img2.png", IMREAD_GRAYSCALE);
//		Mat G2_image3 = imread("D:\\3DOpenSource\\featurepointsdata\\bark-zoomrotation\\ppm\\img3.png", IMREAD_GRAYSCALE);
//		Mat G2_image4 = imread("D:\\3DOpenSource\\featurepointsdata\\bark-zoomrotation\\ppm\\img4.png", IMREAD_GRAYSCALE);
//		Mat G2_image5 = imread("D:\\3DOpenSource\\featurepointsdata\\bark-zoomrotation\\ppm\\img5.png", IMREAD_GRAYSCALE);
//		Mat G2_image6 = imread("D:\\3DOpenSource\\featurepointsdata\\bark-zoomrotation\\ppm\\img6.png", IMREAD_GRAYSCALE);
//		string G2_path = "D:\\3DOpenSource\\featurepointsdata\\bark-zoomrotation\\ppm\\";
//	//Group3 bikes-blur
//		Mat G3_image1 = imread("D:\\3DOpenSource\\featurepointsdata\\bikes-blur\\ppm\\img1.png", IMREAD_GRAYSCALE);
//		Mat G3_image2 = imread("D:\\3DOpenSource\\featurepointsdata\\bikes-blur\\ppm\\img2.png", IMREAD_GRAYSCALE);
//		Mat G3_image3 = imread("D:\\3DOpenSource\\featurepointsdata\\bikes-blur\\ppm\\img3.png", IMREAD_GRAYSCALE);
//		Mat G3_image4 = imread("D:\\3DOpenSource\\featurepointsdata\\bikes-blur\\ppm\\img4.png", IMREAD_GRAYSCALE);
//		Mat G3_image5 = imread("D:\\3DOpenSource\\featurepointsdata\\bikes-blur\\ppm\\img5.png", IMREAD_GRAYSCALE);
//		Mat G3_image6 = imread("D:\\3DOpenSource\\featurepointsdata\\bikes-blur\\ppm\\img6.png", IMREAD_GRAYSCALE);
//		string G3_path = "D:\\3DOpenSource\\featurepointsdata\\bikes-blur\\ppm\\";
//	//Group4 boat-zoomrotation
//		Mat G4_image1 = imread("D:\\3DOpenSource\\featurepointsdata\\boat-zoomrotation\\ppm\\img1.png", IMREAD_GRAYSCALE);
//		Mat G4_image2 = imread("D:\\3DOpenSource\\featurepointsdata\\boat-zoomrotation\\ppm\\img2.png", IMREAD_GRAYSCALE);
//		Mat G4_image3 = imread("D:\\3DOpenSource\\featurepointsdata\\boat-zoomrotation\\ppm\\img3.png", IMREAD_GRAYSCALE);
//		Mat G4_image4 = imread("D:\\3DOpenSource\\featurepointsdata\\boat-zoomrotation\\ppm\\img4.png", IMREAD_GRAYSCALE);
//		Mat G4_image5 = imread("D:\\3DOpenSource\\featurepointsdata\\boat-zoomrotation\\ppm\\img5.png", IMREAD_GRAYSCALE);
//		Mat G4_image6 = imread("D:\\3DOpenSource\\featurepointsdata\\boat-zoomrotation\\ppm\\img6.png", IMREAD_GRAYSCALE);
//		string G4_path = "D:\\3DOpenSource\\featurepointsdata\\boat-zoomrotation\\ppm\\";
//	//Group5 leuven-light
//		Mat G5_image1 = imread("D:\\3DOpenSource\\featurepointsdata\\leuven-light\\ppm\\img1.png", IMREAD_GRAYSCALE);
//		Mat G5_image2 = imread("D:\\3DOpenSource\\featurepointsdata\\leuven-light\\ppm\\img2.png", IMREAD_GRAYSCALE);
//		Mat G5_image3 = imread("D:\\3DOpenSource\\featurepointsdata\\leuven-light\\ppm\\img3.png", IMREAD_GRAYSCALE);
//		Mat G5_image4 = imread("D:\\3DOpenSource\\featurepointsdata\\leuven-light\\ppm\\img4.png", IMREAD_GRAYSCALE);
//		Mat G5_image5 = imread("D:\\3DOpenSource\\featurepointsdata\\leuven-light\\ppm\\img5.png", IMREAD_GRAYSCALE);
//		Mat G5_image6 = imread("D:\\3DOpenSource\\featurepointsdata\\leuven-light\\ppm\\img6.png", IMREAD_GRAYSCALE);
//		string G5_path = "D:\\3DOpenSource\\featurepointsdata\\leuven-light\\ppm\\";
//	//Group6 trees-blur
//		Mat G6_image1 = imread("D:\\3DOpenSource\\featurepointsdata\\trees-blur\\ppm\\img1.png", IMREAD_GRAYSCALE);
//		Mat G6_image2 = imread("D:\\3DOpenSource\\featurepointsdata\\trees-blur\\ppm\\img2.png", IMREAD_GRAYSCALE);
//		Mat G6_image3 = imread("D:\\3DOpenSource\\featurepointsdata\\trees-blur\\ppm\\img3.png", IMREAD_GRAYSCALE);
//		Mat G6_image4 = imread("D:\\3DOpenSource\\featurepointsdata\\trees-blur\\ppm\\img4.png", IMREAD_GRAYSCALE);
//		Mat G6_image5 = imread("D:\\3DOpenSource\\featurepointsdata\\trees-blur\\ppm\\img5.png", IMREAD_GRAYSCALE);
//		Mat G6_image6 = imread("D:\\3DOpenSource\\featurepointsdata\\trees-blur\\ppm\\img6.png", IMREAD_GRAYSCALE);
//		string G6_path = "D:\\3DOpenSource\\featurepointsdata\\trees-blur\\ppm\\";
//	//Group7 ubc-jpegcompression
//		Mat G7_image1 = imread("D:\\3DOpenSource\\featurepointsdata\\ubc-jpegcompression\\ppm\\img1.png", IMREAD_GRAYSCALE);
//		Mat G7_image2 = imread("D:\\3DOpenSource\\featurepointsdata\\ubc-jpegcompression\\ppm\\img2.png", IMREAD_GRAYSCALE);
//		Mat G7_image3 = imread("D:\\3DOpenSource\\featurepointsdata\\ubc-jpegcompression\\ppm\\img3.png", IMREAD_GRAYSCALE);
//		Mat G7_image4 = imread("D:\\3DOpenSource\\featurepointsdata\\ubc-jpegcompression\\ppm\\img4.png", IMREAD_GRAYSCALE);
//		Mat G7_image5 = imread("D:\\3DOpenSource\\featurepointsdata\\ubc-jpegcompression\\ppm\\img5.png", IMREAD_GRAYSCALE);
//		Mat G7_image6 = imread("D:\\3DOpenSource\\featurepointsdata\\ubc-jpegcompression\\ppm\\img6.png", IMREAD_GRAYSCALE);
//		string G7_path = "D:\\3DOpenSource\\featurepointsdata\\ubc-jpegcompression\\ppm\\";
//	//Group8 wall-viewpoint
//		Mat G8_image1 = imread("D:\\3DOpenSource\\featurepointsdata\\wall-viewpoint\\ppm\\img1.png", IMREAD_GRAYSCALE);
//		Mat G8_image2 = imread("D:\\3DOpenSource\\featurepointsdata\\wall-viewpoint\\ppm\\img2.png", IMREAD_GRAYSCALE);
//		Mat G8_image3 = imread("D:\\3DOpenSource\\featurepointsdata\\wall-viewpoint\\ppm\\img3.png", IMREAD_GRAYSCALE);
//		Mat G8_image4 = imread("D:\\3DOpenSource\\featurepointsdata\\wall-viewpoint\\ppm\\img4.png", IMREAD_GRAYSCALE);
//		Mat G8_image5 = imread("D:\\3DOpenSource\\featurepointsdata\\wall-viewpoint\\ppm\\img5.png", IMREAD_GRAYSCALE);
//		Mat G8_image6 = imread("D:\\3DOpenSource\\featurepointsdata\\wall-viewpoint\\ppm\\img6.png", IMREAD_GRAYSCALE);
//		string G8_path = "D:\\3DOpenSource\\featurepointsdata\\wall-viewpoint\\ppm\\";
//		//测试尺度分布与匹配效果
//	//step1:
//	//	Ptr<SIFT> ptrSIFTexactor;
//	//	Ptr<ORB> ptrORBexactor;
//	//	ptrSIFTexactor = cv::SIFT::create(3000);
//	//	ptrORBexactor = cv::ORB::create(3000);
//	//	std::vector<KeyPoint> keyptsift1, keyptsift2, keyptorb1, keyptorb2;
//	//	cv::Mat descrioptorsift1,descrioptorsift2,descrioptorb1,descrioptorb2;
//	//	std::vector<DMatch> siftmath, orbmath;
//	//	ptrSIFTexactor->detectAndCompute(G6_image1, Mat(), keyptsift1, descrioptorsift1, false);
//	//	ptrSIFTexactor->detectAndCompute(G6_image2, Mat(), keyptsift2, descrioptorsift2, false);
//	//	ptrORBexactor->detectAndCompute(G6_image1, Mat(), keyptorb1, descrioptorb1, false);
//	//	ptrORBexactor->detectAndCompute(G6_image2, Mat(), keyptorb2, descrioptorb2, false);
//	////step2:
//	//	string strPathsiftkeypoints = G6_path + "sift_scale" + ".txt";
//	//	string strPathorbkeypoints = G6_path + "orb_scale" + ".txt";
//	//	std::fstream file_scale_sift, file_scale_orb;
//	//	file_scale_sift.open(strPathsiftkeypoints, std::ios::out);
//	//	file_scale_orb.open(strPathorbkeypoints, std::ios::out);
//	////step3:
//	//	for (int i = 0; i < keyptsift1.size(); ++i)
//	//	{
//	//		int nOctave = ((keyptsift1[i].octave >> 8) & 255);
//	//		int ns =log10f(keyptsift1[i].size /2/1.5)/log10f(2) * 3/ 1.6 +nOctave * 3;
//	//		file_scale_sift << keyptsift1[i].pt.x << "," << keyptsift1[i].pt.y << "," <<nOctave <<","<<ns<< endl;
//	//	}
//	//		
//	//	for (int j = 0; j < keyptsift2.size(); ++j)
//	//	{
//	//		int nOctave = ((keyptsift2[j].octave >> 8) & 255);
//	//		int ns=log10f( keyptsift2[j].size/2/1.5) /log10f(2)* 3  / 1.6 + nOctave * 3;
//	//		file_scale_sift << keyptsift2[j].pt.x << "," << keyptsift2[j].pt.y << "," << nOctave<<","<<ns << endl;
//	//	}
//	//		
//	//	for (int i = 0; i < keyptorb1.size(); ++i)
//	//		file_scale_orb << keyptorb1[i].pt.x << "," << keyptorb1[i].pt.y << "," << ((keyptorb1[i].octave>>8)&255) << endl;
//	//	for (int i=0;i<keyptorb2.size();++i)
//	//		file_scale_orb << keyptorb2[i].pt.x << "," << keyptorb2[i].pt.y << "," << ((keyptorb2[i].octave>>8)&255) << endl;
//	//	file_scale_sift.close();
//	//	file_scale_orb.close();
//	//	cv::Mat g6out;
//	//	drawKeypoints(G6_image1,keyptsift1,g6out,Scalar::all(-1),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//	//	imwrite("D:\\3DOpenSource\\featurepointsdata\\trees-blur\\ppm\\out.png",g6out);
//	//	//step4:match the points
//	//	FlannBasedMatcher matcher;
//	//	vector<DMatch> matches1;
//	//	matcher.match(descrioptorsift1, descrioptorsift2, matches1, Mat());
//	//	double max_dist = 0; double min_dist = 100;
//
//	////-- Quick calculation of max and min distances between keypoints
//	//for (int i = 0; i < descrioptorsift1.rows; i++)
//	//{
//	//	double dist = matches1[i].distance;
//	//	if (dist < min_dist) min_dist = dist;
//	//	if (dist > max_dist) max_dist = dist;
//	//}
//
//	//printf("-- Max dist : %f \n", max_dist);
//	//printf("-- Min dist : %f \n", min_dist);
//
//	////-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
//	////-- PS.- radiusMatch can also be used here.
//	//std::vector< DMatch > good_matches;
//
//	//for (int i = 0; i < descrioptorsift1.rows; i++)
//	//{
//	//	if (matches1[i].distance < 2 * min_dist)
//	//	{
//	//		good_matches.push_back(matches1[i]);
//	//	}
//	//}
//
//	////-- Draw only "good" matches
//	//Mat img_matches;
//	//drawMatches(G6_image1, keyptsift1, G6_image2, keyptsift2,
//	//	good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
//	//	vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//
//	////-- Show detected matches
//	////imshow("Good Matches", img_matches);
//
//	//for (int i = 0; i < good_matches.size(); i++)
//	//{
//	//	printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx);
//	//}
//	//Mat imgmatches;
//	//drawMatches(G6_image1, keyptsift1, G6_image2, keyptsift2, matches1, imgmatches, Scalar::all(-1), Scalar::all(-1));
//	//imwrite(G6_path+".JPG", img_matches);
//	////imshow("Matches", imgmatches);
//	//	
//		//test 选择角度测试
//		fstream fileangle;
//		string str45 = G1_path + "45.png";
//		string str90 = G1_path + "90.png";
//		string str135 = G1_path + "135.png";
//		string str180 = G1_path + "180.png";
//		string str225 = G1_path + "225.png";
//		string str270 = G1_path + "270.png";
//		string str315 = G1_path + "315.png";
//		cv::Mat im45, im90, im135, im180, im225, im270, im315;
//		rotate_arbitrarily_angle(G1_image2, im45, 45);
//		rotate_arbitrarily_angle(G1_image2, im90, 90);
//		rotate_arbitrarily_angle(G1_image2, im135, 135);
//		rotate_arbitrarily_angle(G1_image2, im180, 180);
//		rotate_arbitrarily_angle(G1_image2, im225, 225);
//		rotate_arbitrarily_angle(G1_image2, im270, 270);
//		rotate_arbitrarily_angle(G1_image2, im315, 315);
//		imwrite(str45, im45);
//		imwrite(str90, im90);
//		imwrite(str135, im135);
//		imwrite(str180, im180);
//		imwrite(str225, im225);
//		imwrite(str270, im270);
//		imwrite(str315, im315);
//		cv::Mat m45=imread(str45, IMREAD_GRAYSCALE);
//		cv::Mat m90 = imread(str90, IMREAD_GRAYSCALE);
//		cv::Mat m135 = imread(str135, IMREAD_GRAYSCALE);
//		cv::Mat m180 = imread(str180, IMREAD_GRAYSCALE);
//		cv::Mat m225 = imread(str225, IMREAD_GRAYSCALE);
//		cv::Mat m270 = imread(str270, IMREAD_GRAYSCALE);
//		cv::Mat m315 = imread(str315, IMREAD_GRAYSCALE);
//		std::vector<KeyPoint> k1, k2;
//		std::vector<DMatch> mathcees;
//		bool bRansac = true;
//		bool bProvidepoints = false;
//		string strKeytype = "ORB";
//		string  strjiaodufile = G1_path + strKeytype+"angle2.txt";
//		fileangle.open(strjiaodufile, std::ios::out);
//	
//		RunDetectAndMatchTest(strKeytype, G1_image2, m45, k1, k2, mathcees, bRansac,bProvidepoints);
//		fileangle << "45" << "," << k1.size() << "," << k2.size() << "," << mathcees.size() << "," << float(2 * mathcees.size()) / (k1.size() + k2.size()) << "," << bRansac<<bProvidepoints << endl;
//		RunDetectAndMatchTest(strKeytype, G1_image2, m90, k1, k2, mathcees, bRansac,bProvidepoints);
//		fileangle << "90" << "," << k1.size() << "," << k2.size() << "," << mathcees.size() << "," << float(2 * mathcees.size()) / (k1.size() + k2.size()) << "," << bRansac<<bProvidepoints << endl;
//		RunDetectAndMatchTest(strKeytype, G1_image2, m135, k1, k2, mathcees, bRansac,bProvidepoints);
//		fileangle << "135" << "," << k1.size() << "," << k2.size() << "," << mathcees.size() << "," << float(2 * mathcees.size()) / (k1.size() + k2.size()) << "," << bRansac<<bProvidepoints << endl;
//		RunDetectAndMatchTest(strKeytype, G1_image2, m180, k1, k2, mathcees, bRansac,bProvidepoints);
//		fileangle << "180" << "," << k1.size() << "," << k2.size() << "," << mathcees.size() << "," << float(2 * mathcees.size()) / (k1.size() + k2.size()) << "," << bRansac<<bProvidepoints << endl;
//		RunDetectAndMatchTest(strKeytype, G1_image2, m225, k1, k2, mathcees, bRansac,bProvidepoints);
//		fileangle << "225" << "," << k1.size() << "," << k2.size() << "," << mathcees.size() << "," << float(2 * mathcees.size()) / (k1.size() + k2.size()) << "," << bRansac<<bProvidepoints << endl;
//		RunDetectAndMatchTest(strKeytype, G1_image2, m270, k1, k2, mathcees, bRansac,bProvidepoints);
//		fileangle << "270" << "," << k1.size() << "," << k2.size() << "," << mathcees.size() << "," << float(2 * mathcees.size()) / (k1.size() + k2.size()) << "," << bRansac<<bProvidepoints << endl;
//		RunDetectAndMatchTest(strKeytype, G1_image2, m315, k1, k2, mathcees, bRansac,bProvidepoints);
//		fileangle << "315" << "," << k1.size() << "," << k2.size() << "," << mathcees.size() << "," << float(2 * mathcees.size()) / (k1.size() + k2.size()) << "," << bRansac<<bProvidepoints << endl;
//		bRansac=false,bProvidepoints = false;
//		RunDetectAndMatchTest(strKeytype, G1_image2, m45, k1, k2, mathcees, bRansac,bProvidepoints);
//		fileangle << "45" << "," << k1.size() << "," << k2.size() << "," << mathcees.size() << "," << float(2 * mathcees.size()) / (k1.size() + k2.size()) << "," << bRansac<<bProvidepoints << endl;
//		RunDetectAndMatchTest(strKeytype, G1_image2, m90, k1, k2, mathcees, bRansac,bProvidepoints);
//		fileangle << "90" << "," << k1.size() << "," << k2.size() << "," << mathcees.size() << "," << float(2 * mathcees.size()) / (k1.size() + k2.size()) << "," << bRansac<<bProvidepoints << endl;
//		RunDetectAndMatchTest(strKeytype, G1_image2, m135, k1, k2, mathcees, bRansac,bProvidepoints);
//		fileangle << "135" << "," << k1.size() << "," << k2.size() << "," << mathcees.size() << "," << float(2 * mathcees.size()) / (k1.size() + k2.size()) << "," << bRansac<<bProvidepoints << endl;
//		RunDetectAndMatchTest(strKeytype, G1_image2, m180, k1, k2, mathcees, bRansac,bProvidepoints);
//		fileangle << "180" << "," << k1.size() << "," << k2.size() << "," << mathcees.size() << "," << float(2 * mathcees.size()) / (k1.size() + k2.size()) << "," << bRansac<<bProvidepoints << endl;
//		RunDetectAndMatchTest(strKeytype, G1_image2, m225, k1, k2, mathcees, bRansac,bProvidepoints);
//		fileangle << "225" << "," << k1.size() << "," << k2.size() << "," << mathcees.size() << "," << float(2 * mathcees.size()) / (k1.size() + k2.size()) << "," << bRansac<<bProvidepoints << endl;
//		RunDetectAndMatchTest(strKeytype, G1_image2, m270, k1, k2, mathcees, bRansac,bProvidepoints);
//		fileangle << "270" << "," << k1.size() << "," << k2.size() << "," << mathcees.size() << "," << float(2 * mathcees.size()) / (k1.size() + k2.size()) << "," << bRansac<<bProvidepoints << endl;
//		RunDetectAndMatchTest(strKeytype, G1_image2, m315, k1, k2, mathcees, bRansac,bProvidepoints);
//		fileangle << "315" << "," << k1.size() << "," << k2.size() << "," << mathcees.size() << "," << float(2 * mathcees.size()) / (k1.size() + k2.size()) << "," << bRansac << endl;
//		fileangle.close();
//		//test ORB 速度和sift速度
//		{
//			Ptr<SIFT>  pexSifte;
//			pexSifte = cv::SIFT::create();
//			std::vector<KeyPoint> vk1, vk2, vk3, vk4, vk5, vk6, vk7, vk8;
//			pexSifte->detect(G1_image1, vk1);
//			pexSifte->detect(G2_image1, vk2);
//			pexSifte->detect(G3_image1, vk3);
//			pexSifte->detect(G4_image1, vk4);
//			pexSifte->detect(G5_image1, vk5);
//			pexSifte->detect(G6_image1, vk6);
//			pexSifte->detect(G7_image1, vk7);
//			pexSifte->detect(G8_image1, vk8);
//			std::cout << vk1.size() << std::endl;
//			std::cout << vk2.size() << std::endl;
//			std::cout << vk3.size() << std::endl;
//			std::cout << vk4.size() << std::endl;
//			std::cout << vk5.size() << std::endl;
//			std::cout << vk6.size() << std::endl;
//			std::cout << vk7.size() << std::endl;
//			std::cout << vk8.size() << std::endl;
//		
//		}
//		
//
//
//
//
//
//
//
//		char * typej = const_cast<char*>("SIFT");
//		char * typem = const_cast<char*>("ORB");
//		myTime *mysifttime = new myTime(typej);
//		myTime * myorbtime = new myTime(typem);
//		/*cout << "G1 test start..........................." << endl;
//		mysifttime->start();
//		RunExactFeatureTimeTest("SIFT", G1_image1);
//		RunExactFeatureTimeTest("SIFT", G1_image2);
//		RunExactFeatureTimeTest("SIFT", G1_image3);
//		RunExactFeatureTimeTest("SIFT", G1_image4);
//		RunExactFeatureTimeTest("SIFT", G1_image5);
//		RunExactFeatureTimeTest("SIFT", G1_image6);
//		mysifttime->stop();
//		mysifttime->start();
//		cout << "G2 test start..........................." << endl;
//		RunExactFeatureTimeTest("SIFT", G2_image1);
//		RunExactFeatureTimeTest("SIFT", G2_image2);
//		RunExactFeatureTimeTest("SIFT", G2_image3);
//		RunExactFeatureTimeTest("SIFT", G2_image4);
//		RunExactFeatureTimeTest("SIFT", G2_image5);
//		RunExactFeatureTimeTest("SIFT", G2_image6);
//		mysifttime->stop();
//		mysifttime->start();
//		cout << "G3 test start..........................." << endl;
//		RunExactFeatureTimeTest("SIFT", G3_image1);
//		RunExactFeatureTimeTest("SIFT", G3_image2);
//		RunExactFeatureTimeTest("SIFT", G3_image3);
//		RunExactFeatureTimeTest("SIFT", G3_image4);
//		RunExactFeatureTimeTest("SIFT", G3_image5);
//		RunExactFeatureTimeTest("SIFT", G3_image6);
//		mysifttime->stop();
//		mysifttime->start();
//		cout << "G4 test start..........................." << endl;
//		RunExactFeatureTimeTest("SIFT", G4_image1);
//		RunExactFeatureTimeTest("SIFT", G4_image2);
//		RunExactFeatureTimeTest("SIFT", G4_image3);
//		RunExactFeatureTimeTest("SIFT", G4_image4);
//		RunExactFeatureTimeTest("SIFT", G4_image5);
//		RunExactFeatureTimeTest("SIFT", G4_image6);
//		mysifttime->stop();
//		mysifttime->start();
//		cout << "G5 test start..........................." << endl;
//		RunExactFeatureTimeTest("SIFT", G5_image1);
//		RunExactFeatureTimeTest("SIFT", G5_image2);
//		RunExactFeatureTimeTest("SIFT", G5_image3);
//		RunExactFeatureTimeTest("SIFT", G5_image4);
//		RunExactFeatureTimeTest("SIFT", G5_image5);
//		RunExactFeatureTimeTest("SIFT", G5_image6);
//		mysifttime->stop();
//		mysifttime->start();
//		cout << "G6 test start..........................." << endl;
//		RunExactFeatureTimeTest("SIFT", G6_image1);
//		RunExactFeatureTimeTest("SIFT", G6_image2);
//		RunExactFeatureTimeTest("SIFT", G6_image3);
//		RunExactFeatureTimeTest("SIFT", G6_image4);
//		RunExactFeatureTimeTest("SIFT", G6_image5);
//		RunExactFeatureTimeTest("SIFT", G6_image6);
//		mysifttime->stop();
//		mysifttime->start();
//		cout << "G7 test start..........................." << endl;
//		RunExactFeatureTimeTest("SIFT", G7_image1);
//		RunExactFeatureTimeTest("SIFT", G7_image2);
//		RunExactFeatureTimeTest("SIFT", G7_image3);
//		RunExactFeatureTimeTest("SIFT", G7_image4);
//		RunExactFeatureTimeTest("SIFT", G7_image5);
//		RunExactFeatureTimeTest("SIFT", G7_image6);
//		mysifttime->stop();
//		mysifttime->start();
//		cout << "G8 test start..........................." << endl;
//		RunExactFeatureTimeTest("SIFT", G8_image1);
//		RunExactFeatureTimeTest("SIFT", G8_image2);
//		RunExactFeatureTimeTest("SIFT", G8_image3);
//		RunExactFeatureTimeTest("SIFT", G8_image4);
//		RunExactFeatureTimeTest("SIFT", G8_image5);
//		RunExactFeatureTimeTest("SIFT", G8_image6);
//		mysifttime->stop();
//		cout << "-------------ORB--------------" << endl;
//		cout << " G1 test start..........................." << endl;
//		myorbtime->start();
//		RunExactFeatureTimeTest("ORB", G1_image1);
//		RunExactFeatureTimeTest("ORB", G1_image2);
//		RunExactFeatureTimeTest("ORB", G1_image3);
//		RunExactFeatureTimeTest("ORB", G1_image4);
//		RunExactFeatureTimeTest("ORB", G1_image5);
//		RunExactFeatureTimeTest("ORB", G1_image6);
//		myorbtime->stop();
//		cout << " G2 test start..........................." << endl;
//		myorbtime->start();
//		RunExactFeatureTimeTest("ORB", G2_image1);
//		RunExactFeatureTimeTest("ORB", G2_image2);
//		RunExactFeatureTimeTest("ORB", G2_image3);
//		RunExactFeatureTimeTest("ORB", G2_image4);
//		RunExactFeatureTimeTest("ORB", G2_image5);
//		RunExactFeatureTimeTest("ORB", G2_image6);
//		myorbtime->stop();
//		cout << " G3 test start..........................." << endl;
//		myorbtime->start();
//		RunExactFeatureTimeTest("ORB", G3_image1);
//		RunExactFeatureTimeTest("ORB", G3_image2);
//		RunExactFeatureTimeTest("ORB", G3_image3);
//		RunExactFeatureTimeTest("ORB", G3_image4);
//		RunExactFeatureTimeTest("ORB", G3_image5);
//		RunExactFeatureTimeTest("ORB", G3_image6);
//		myorbtime->stop();
//		cout << " G4 test start..........................." << endl;
//		myorbtime->start();
//		RunExactFeatureTimeTest("ORB", G4_image1);
//		RunExactFeatureTimeTest("ORB", G4_image2);
//		RunExactFeatureTimeTest("ORB", G4_image3);
//		RunExactFeatureTimeTest("ORB", G4_image4);
//		RunExactFeatureTimeTest("ORB", G4_image5);
//		RunExactFeatureTimeTest("ORB", G4_image6);
//		myorbtime->stop();
//		cout << " G5 test start..........................." << endl;
//		myorbtime->start();
//		RunExactFeatureTimeTest("ORB", G5_image1);
//		RunExactFeatureTimeTest("ORB", G5_image2);
//		RunExactFeatureTimeTest("ORB", G5_image3);
//		RunExactFeatureTimeTest("ORB", G5_image4);
//		RunExactFeatureTimeTest("ORB", G5_image5);
//		RunExactFeatureTimeTest("ORB", G5_image6);
//		myorbtime->stop();
//		cout << " G6 test start..........................." << endl;
//		myorbtime->start();
//		RunExactFeatureTimeTest("ORB", G6_image1);
//		RunExactFeatureTimeTest("ORB", G6_image2);
//		RunExactFeatureTimeTest("ORB", G6_image3);
//		RunExactFeatureTimeTest("ORB", G6_image4);
//		RunExactFeatureTimeTest("ORB", G6_image5);
//		RunExactFeatureTimeTest("ORB", G6_image6);
//		myorbtime->stop();
//		cout << " G7 test start..........................." << endl;
//		myorbtime->start();
//		RunExactFeatureTimeTest("ORB", G7_image1);
//		RunExactFeatureTimeTest("ORB", G7_image2);
//		RunExactFeatureTimeTest("ORB", G7_image3);
//		RunExactFeatureTimeTest("ORB", G7_image4);
//		RunExactFeatureTimeTest("ORB", G7_image5);
//		RunExactFeatureTimeTest("ORB", G7_image6);
//		myorbtime->stop();
//		cout << " G8 test start..........................." << endl;
//		myorbtime->start();
//		RunExactFeatureTimeTest("ORB", G8_image1);
//		RunExactFeatureTimeTest("ORB", G8_image2);
//		RunExactFeatureTimeTest("ORB", G8_image3);
//		RunExactFeatureTimeTest("ORB", G8_image4);
//		RunExactFeatureTimeTest("ORB", G8_image5);
//		RunExactFeatureTimeTest("ORB", G8_image6);
//		myorbtime->stop();*/
//		cout << "test end.................................." << endl;
//
//
////	std::vector<string> strImageLists,dstImageLists;
////	for (int i=1;i<=6;++i)
////	{
////		string strFirst = "D:\\3DOpenSource\\featurepointsdata\\wall\\ppm\\img";
////		string strEnd = ".ppm";
////		string strEndOut = ".png";
////		string strInput = strFirst + std::to_string(i) + strEnd;
////		string strOutput = strFirst + std::to_string(i) + strEndOut;
////		strImageLists.push_back(strInput);
////		dstImageLists.push_back(strOutput);
////	}
////	//ConvertPPMToPNG(strImageLists, dstImageLists);
////	//判断图像是否能正常打开
////	if (!image1.data || !image2.data)
////	{
////		std::cout << "Error reading images!!" << endl;
////		return -1;
////	}
////	//匹配结果
////	std::vector<cv::DMatch> matches;
////	std::vector<cv::KeyPoint> keypoints1, keypoints2, keypoints3, keypoints4, keypoints5, keypoints6, keypoints7,keypoints8;
////	std::vector<cv::Point2f>  keypoints1_, keypoints2_;
////	cv::Mat descriptor3, descriptor4;
////	//测试特征提取时间
////
////	//创建稳健匹配器
////	RobustMatcher rmatcher(cv::SIFT::create(1000));
////	cv::Mat fundamental = rmatcher.match(image1, image2,
////		matches, keypoints1, keypoints2);
////	cv::Mat imageMatches;
////	cv::drawMatches(image1, keypoints1,  // 1st image and its keypoints
////		image2, keypoints2,  // 2nd image and its keypoints
////		matches,			// the matches
////		imageMatches,		// the image produced
////		cv::Scalar(-1),  // color of the lines
////		cv::Scalar(-1)  // color of the keypoints
////	);
////	//cv::imshow("Matches", imageMatches);
//////cv::imwrite("D:\\data\\south-building\\images\\contactRansac.JPG", imageMatches);
////	//测试部分1
////	RunExactFeatureTimeTest("SIFT", image1);
////	//创建sift提取器
////	Ptr<SIFT> pexactor1 = cv::SIFT::create(5000);
////	Ptr<SIFT> pexactor2 = cv::SIFT::create(5000);
////	Ptr<SIFT> pexactor3 = cv::SIFT::create();
////	Ptr<SIFT> pexactor4 = cv::SIFT::create();
////	Ptr<SIFT> pexactor5;
////	//提取sift特征
////	pexactor1->detectAndCompute(image1, Mat(), keypoints3, descriptor3,false);
////	pexactor2->detectAndCompute(image2, Mat(), keypoints4, descriptor4, false);
////
////	//测试部分2
////	{
////		RunMatchTest("FLANN", image1, image2, keypoints3, keypoints4, descriptor3, descriptor4);
////	
////	
////	}
////
////	//测试部分3
////	// Convert keypoints into Point2f	
////	//转换特征点类型
////	std::vector<cv::Point2f> points1, points2;
////	for (std::vector<cv::DMatch>::const_iterator it = matches.begin();
////		it != matches.end(); ++it) {
////
////		// Get the position of left keypoints
////		float x = keypoints1[it->queryIdx].pt.x;
////		float y = keypoints1[it->queryIdx].pt.y;
////		points1.push_back(keypoints1[it->queryIdx].pt);
////		cv::circle(image1, cv::Point(x, y), 3, cv::Scalar(255, 255, 255), 3);
////		// Get the position of right keypoints
////		x = keypoints2[it->trainIdx].pt.x;
////		y = keypoints2[it->trainIdx].pt.y;
////		cv::circle(image2, cv::Point(x, y), 3, cv::Scalar(255, 255, 255), 3);
////		points2.push_back(keypoints2[it->trainIdx].pt);
////	}
////	//先进行ransac提纯后在进行最小二乘计算单应矩阵
////	
////	BFMatcher bFmatcher;
////	vector<DMatch> matches_Repeat,matches_refine;
////	bFmatcher.match(descriptor3, descriptor4, matches_Repeat);
////	RansacTest(matches_Repeat, keypoints3, keypoints4, matches_refine);
////	std::vector<cv::Point2f> points3, points4,points3_refine,points4_refine;
////	for (std::vector<cv::DMatch>::const_iterator it = matches_Repeat.begin();
////		it != matches_Repeat.end(); ++it) {
////
////		// Get the position of left keypoints
////		float x = keypoints3[it->queryIdx].pt.x;
////		float y = keypoints3[it->queryIdx].pt.y;
////		points3.push_back(keypoints3[it->queryIdx].pt);
////		cv::circle(image1, cv::Point(x, y), 3, cv::Scalar(255, 255, 255), 3);
////		// Get the position of right keypoints
////		x = keypoints4[it->trainIdx].pt.x;
////		y = keypoints4[it->trainIdx].pt.y;
////		cv::circle(image2, cv::Point(x, y), 3, cv::Scalar(255, 255, 255), 3);
////		points4.push_back(keypoints2[it->trainIdx].pt);
////	}
////	for (std::vector<cv::DMatch>::const_iterator it = matches_refine.begin();
////		it != matches_refine.end(); ++it) {
////
////		// Get the position of left keypoints
////		float x = keypoints3[it->queryIdx].pt.x;
////		float y = keypoints3[it->queryIdx].pt.y;
////		points3_refine.push_back(keypoints3[it->queryIdx].pt);
////		cv::circle(image1, cv::Point(x, y), 3, cv::Scalar(255, 255, 255), 3);
////		// Get the position of right keypoints
////		x = keypoints4[it->trainIdx].pt.x;
////		y = keypoints4[it->trainIdx].pt.y;
////		cv::circle(image2, cv::Point(x, y), 3, cv::Scalar(255, 255, 255), 3);
////		points4_refine.push_back(keypoints2[it->trainIdx].pt);
////	}
////	std::vector<uchar> inliers(points4.size(), 0);
////	cv::Mat  Findmeatal = cv::findFundamentalMat(points3, points4,cv::RANSAC);
////	cv::Mat HomoGraphy = cv::findHomography(points3, points4, cv::RANSAC);
////	cv::Mat HomoGraphy1 = cv::findHomography(points3, points4, cv::LMEDS);
////	cv::Mat HomoGraphy2 = cv::findHomography(points3, points4, cv::RHO);
////	cv::Mat HomoGraphy3 = cv::findHomography(points3_refine, points4_refine, cv::LMEDS);
//
//	//testHomgraphy(points3, points4, HomoGraphy);
//	//Group1
//	cv::Mat G1_h12=(cv::Mat_<double>(3,3)<<8.7976964e-01,3.1245438e-01,-3.9430589e+01,-1.8389418e-01,9.3847198e-01,1.5315784e+02,
//   1.9641425e-04 ,- 1.6015275e-05  , 1.0000000e+00 );
//	cv::Mat G1_h13 = (cv::Mat_<double>(3, 3) << 7.6285898e-01, -2.9922929e-01, 2.2567123e+02
//		, 3.3443473e-01, 1.0143901e+00, -7.6999973e+01
//		, 3.4663091e-04, -1.4364524e-05, 1.0000000e+00);
//	cv::Mat G1_h14 = (cv::Mat_<double>(3, 3) << 6.6378505e-01  , 6.8003334e-01 ,- 3.1230335e+01
//		,- 1.4495500e-01  , 9.7128304e-01  , 1.4877420e+02
//		,4.2518504e-04, - 1.3930359e-05  , 1.0000000e+00);
//	cv::Mat G1_h15 = (cv::Mat_<double>(3, 3) << 6.2544644e-01  , 5.7759174e-02  , 2.2201217e+02
//		,2.2240536e-01 ,  1.1652147e+00 ,- 2.5605611e+01
//		,4.9212545e-04 ,- 3.6542424e-05 ,  1.0000000e+00);
//	cv::Mat G1_h16 = (cv::Mat_<double>(3, 3) << 4.2714590e-01 ,- 6.7181765e-01 , 4.5361534e+02
//		,4.4106579e-01  , 1.0133230e+00 ,- 4.6534569e+01
//		,5.1887712e-04 ,- 7.8853731e-05 , 1.0000000e+00);
//	//Group2
//	cv::Mat G2_h12 = (cv::Mat_<double>(3, 3) << 0.7022029025774007,  0.4313737491020563 ,- 127.94661199701689
//		,- 0.42757325092889575 , 0.6997834349758094 , 201.26193857481698
//		,4.083733373964227E-6 , 1.5076445750988132E-5 , 1.0
//		);
//	cv::Mat G2_h13 = (cv::Mat_<double>(3, 3) << -0.48367041358997964 ,- 0.2472935325077872,  870.2215120216712
//		,0.29085746679198893 ,- 0.45733473891783305 , 396.1604918833091
//		,- 3.578663704630333E-6,  6.880007548843957E-5 , 1.0);
//	cv::Mat G2_h14 = (cv::Mat_<double>(3, 3) << -0.20381418476462312 , 0.3510201271914591 , 247.1085214229702
//		,- 0.3499531830464912 ,- 0.1975486500576974 , 466.54576370699766
//		,- 1.5735788289619667E-5 , 1.0242951905091244E-5 ,1.0
//		);
//	cv::Mat G2_h15 = (cv::Mat_<double>(3, 3) << 0.30558415717792214 , 0.12841186681168829 , 200.94588793078017
//		,- 0.12861248979242065 , 0.3067557133397112  ,133.77000196887894
//		,2.782320090398499E-6 , 5.770764104061954E-6 , 1.0);
//	cv::Mat G2_h16 = (cv::Mat_<double>(3, 3) << -0.23047631546234373 ,- 0.10655686701035443 , 583.3200507850402
//		,0.11269946585180685 ,- 0.20718914340861153  ,355.2381263740649
//		,- 3.580280012615393E-5 ,3.2283960511548054E-5 , 1.0);
//	//Group3
//	cv::Mat G3_h12 = (cv::Mat_<double>(3, 3) << 1.0107879e+00 ,  8.2814684e-03 ,  1.8576800e+01
//		,- 4.9128885e-03  , 1.0148779e+00 ,- 2.8851517e+01
//		,- 1.9166087e-06 ,  8.1537620e-06  , 1.0000000e+00
//		);
//	cv::Mat G3_h13 = (cv::Mat_<double>(3, 3) << 1.0129406e+00 ,  7.0258059e-03, - 3.5409366e+00
//		,- 4.3550970e-03  , 1.0183920e+00, - 3.2761060e+01
//		,- 2.9227621e-06 ,  9.0460793e-06  , 1.0000000e+00);
//	cv::Mat G3_h14 = (cv::Mat_<double>(3, 3) << 1.0201734e+00  , 1.3125949e-02 ,- 1.0048666e+01
//		,- 7.9558939e-03 ,  1.0253060e+00 ,- 4.3000272e+01
//		,- 2.2467584e-06 ,  1.2471581e-05  , 1.0000000e+00);
//	cv::Mat G3_h15 = (cv::Mat_<double>(3, 3) << 1.0261529e+00  , 1.2319444e-02,- 8.5197497e+00
//		,- 8.3147838e-03 ,  1.0311644e+00 ,- 4.1319031e+01
//		,- 9.1200792e-08  , 1.0876260e-05 ,  1.0000000e+00);
//	cv::Mat G3_h16 = (cv::Mat_<double>(3, 3) << 1.0427236e+00  , 1.2359858e-02 ,- 1.6974167e+01
//		,- 4.2238744e-03  , 1.0353397e+00 ,- 4.5312478e+01
//		,1.2020516e-05 ,  8.2950327e-06  , 1.0000000e+00);
//	//Group4
//	cv::Mat G4_h12 = (cv::Mat_<double>(3, 3) << 8.5828552e-01 ,  2.1564369e-01  , 9.9101418e+00
//		,- 2.1158440e-01 ,  8.5876360e-01  , 1.3047838e+02
//		,2.0702435e-06 ,  1.2886110e-06  , 1.0000000e+00);
//	cv::Mat G4_h13 = (cv::Mat_<double>(3, 3) << 5.6887079e-01 ,  4.6997572e-01 ,  2.5515642e+01
//		,- 4.6783159e-01  , 5.6548769e-01  , 3.4819925e+02
//		,6.4697420e-06, - 1.1704138e-06  , 1.0000000e+00);
//	cv::Mat G4_h14 = (cv::Mat_<double>(3, 3) << 1.0016637e-01 ,  5.2319717e-01  , 2.0587932e+02
//		,- 5.2345249e-01 , 8.7390786e-02  , 5.3454522e+02
//		,9.4931475e-06 ,- 9.8296917e-06 ,  1.0000000e+00);
//	cv::Mat G4_h15 = (cv::Mat_<double>(3, 3) << 4.2310823e-01 ,- 6.0670438e-02  , 2.6635003e+02
//		,6.2730152e-02  , 4.1652096e-01  , 1.7460201e+02
//		,1.5812849e-05 ,- 1.4368783e-05  , 1.0000000e+00);
//	cv::Mat G4_h16 = (cv::Mat_<double>(3, 3) << 2.9992872e-01  , 2.2821975e-01  , 2.2930182e+02
//		,- 2.3832758e-01  , 2.4564042e-01 ,  3.6767399e+02
//		,9.9064973e-05 ,- 5.8498673e-05 ,  1.0000000e+00);
//	//Group5
//	cv::Mat G5_h12 = (cv::Mat_<double>(3, 3) << 5.7783232e-01 ,- 1.8122966e-04 ,  2.8225664e+00
//		,2.2114401e-03 ,  5.7937539e-01 ,- 1.7879175e+00
//		,- 2.3911512e-06 ,  2.9032886e-06 ,  5.7865196e-01);
//	cv::Mat G5_h13 = (cv::Mat_<double>(3, 3) << 5.7386650e-01  , 3.2769965e-03  , 2.8696063e+00
//		,- 4.2640197e-04 ,  5.7654920e-01 ,- 2.6477989e+00
//		,- 3.5986087e-06 ,  5.4939501e-06 ,  5.7489565e-01);
//	cv::Mat G5_h14 = (cv::Mat_<double>(3, 3) << 5.7494804e-01 ,  2.7800742e-03  , 4.9723266e+00
//		,1.7588927e-03  , 5.7873002e-01 ,- 5.4767862e+00
//		,- 4.9951367e-06 ,  8.0784390e-06 ,  5.7639952e-01);
//	cv::Mat G5_h15 = (cv::Mat_<double>(3, 3) << -5.7780461e-01 ,- 7.4653534e-03, - 1.8849466e-01
//		,8.4414658e-04 ,- 5.8051698e-01  , 4.5205045e+00
//		,2.0525034e-06 ,- 1.0250081e-05 ,- 5.7604815e-01);
//	cv::Mat G5_h16 = (cv::Mat_<double>(3, 3) << 5.8695833e-01 ,  6.2763397e-03  , 1.3078972e+00
//		,1.9788878e-03  , 5.8978058e-01 ,- 9.5598967e+00
//		,- 1.6508045e-06  , 1.3162429e-05  , 5.8394502e-01);
//	//Group6
//	cv::Mat G6_h12 = (cv::Mat_<double>(3, 3) << 0.9912089374539974,  0.04561277689934186 , 16.430575146746467
//		,- 0.047962294548037 , 0.9957951079956443  ,17.73539383094122
//		,- 8.73992330142086E-6 , 1.1499680166976166E-6 , 1.0);
//	cv::Mat G6_h13 = (cv::Mat_<double>(3, 3) << 0.9905632952782818 , 0.043347675611079745  ,8.236464788207753
//		,- 0.04702626594025669 , 0.9988185365751873 , 17.373870105550285
//		,- 1.5932518558327435E-5 , 2.893846251213057E-6 , 1.0);
//	cv::Mat G6_h14 = (cv::Mat_<double>(3, 3) << 1.0263805284648657 , 0.04713298536155905 ,- 16.259771505387544
//		,- 0.038844062111074 , 1.0188531347224243  ,0.6449843282481993
//		,7.567223321612053E-6 , 3.665043946826549E-6 , 1.0);
//	cv::Mat G6_h15 = (cv::Mat_<double>(3, 3) << 1.0222521389207018 , 0.04749404190465927 ,- 25.968189130651552
//		,- 0.04681492541525158 , 1.0157636596278663  ,10.560803317308023
//		,3.244887964210479E-6 ,- 3.4221108923317904E-6 , 1.0);
//	cv::Mat G6_h16 = (cv::Mat_<double>(3, 3) << 1.0364619265098058 , 0.054448231785273325 ,- 26.573496349036247
//		,- 0.050205448575418116 , 1.022285037696358 , 9.089883116763504
//		,1.0110609732276445E-5 ,- 6.405721835180334E-6 , 1.0);
//	//Group7
//	cv::Mat G7_h12 = (cv::Mat_<double>(3, 3) << 1, 0 ,0
//		,0 ,1 ,0
//		,0, 0 ,1);
//	cv::Mat G7_h13 = (cv::Mat_<double>(3, 3) << 1, 0 ,0
//		,0 ,1 ,0
//		,0, 0 ,1);
//	cv::Mat G7_h14 = (cv::Mat_<double>(3, 3) << 1, 0 ,0
//		,0 ,1 ,0
//		,0, 0, 1);
//	cv::Mat G7_h15 = (cv::Mat_<double>(3, 3) << 1 ,0 ,0
//		,0, 1, 0
//		,0 ,0, 1);
//	cv::Mat G7_h16 = (cv::Mat_<double>(3, 3) << 1 ,0 ,0
//		,0 ,1 ,0
//		,0 ,0, 1);
//	//Group8
//	cv::Mat G8_h12 = (cv::Mat_<double>(3, 3) << 0.7882767153207999 , 0.010905680735846527 , 28.170495497465602
//		,- 0.02537010994777608 , 0.9232684706505401 , 44.20085016989556
//		,- 1.1457814415224265E-4  ,1.288160474307972E-5 , 1.0);
//	cv::Mat G8_h13 = (cv::Mat_<double>(3, 3) << 0.6682947339156113,  0.018344318347851395,  39.51916188173466
//		,- 0.04902747132995888 , 0.8935012492790394 , 61.81007229702091
//		,- 1.8999645773011534E-4 , 2.069199620253009E-6 , 1.0);
//	cv::Mat G8_h14 = (cv::Mat_<double>(3, 3) << 0.5487967233294201 , 0.015245351406439072 , 65.03321744618472
//		,- 0.06274161341697451 , 0.8804280211603792 , 105.39150873162244
//		,- 2.469232356641658E-4 , 1.6209582458142305E-5  ,1.0);
//	cv::Mat G8_h15 = (cv::Mat_<double>(3, 3) << 0.4133591554597665 , 0.026091530324690363 , 61.728731455568294
//		,- 0.08735789108803846 , 0.8660455372469953 , 95.35388347437842
//		,- 3.3657799873241785E-4 ,- 8.590195344076651E-6 , 1.0);
//	cv::Mat G8_h16 = (cv::Mat_<double>(3, 3) << 0.25816310023931976 , 0.028122203548214684  ,122.77193310808889
//		,- 0.09827340705979042 , 0.9034280861806072  ,87.97366097395911
//		,- 4.096870682438919E-4 , 8.67484264796887E-7 , 1.0);
//	
//	//测试ORB算法的重现率和正确计数
//	std::vector<InformationReapet> vecInoformation;
//	std::vector<cv::KeyPoint> keypoints_orb_left, keypoints_orb_right;
//	{
//		cv::Ptr<SIFT>  pORBexactor;
//		pORBexactor = cv::SIFT::create(5000);
//		InformationReapet tempInformation;
//		float reapetality = 0.0;
//		int nCrosCount = 0;
//		string keyType = "SIFT";
//	   //test G1
//		RunExactDetection(G1_image1, G1_image2,keypoints_orb_left,keypoints_orb_right,G1_h12,reapetality,nCrosCount,pORBexactor);
//		tempInformation.groupName = "G1";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G1_image1, keypoints_orb_left, G1_path + keyType+ "img1_key.png");
//		DrawSaveKeypoints(G1_image2, keypoints_orb_right, G1_path + keyType + "img2_key.png");
//		RunExactDetection(G1_image1, G1_image3, keypoints_orb_left, keypoints_orb_right, G1_h13, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G1";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G1_image3, keypoints_orb_right, G1_path + keyType + "img3_key.png");
//		RunExactDetection(G1_image1, G1_image4, keypoints_orb_left, keypoints_orb_right, G1_h14, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G1";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G1_image4, keypoints_orb_right, G1_path + keyType + "img4_key.png");
//		RunExactDetection(G1_image1, G1_image5, keypoints_orb_left, keypoints_orb_right, G1_h15, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G1";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G1_image5, keypoints_orb_right, G1_path + keyType + "img5_key.png");
//		RunExactDetection(G1_image1, G1_image6, keypoints_orb_left, keypoints_orb_right, G1_h16, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G1";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G1_image6, keypoints_orb_right, G1_path + keyType + "img6_key.png");
//		//test G2
//		cout << "G2_start........" << endl;
//		RunExactDetection(G2_image1, G2_image2, keypoints_orb_left, keypoints_orb_right, G2_h12, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G2";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G2_image1, keypoints_orb_left, G2_path + keyType + "img1_key.png");
//		DrawSaveKeypoints(G2_image2, keypoints_orb_right, G2_path + keyType + "img2_key.png");
//		RunExactDetection(G2_image1, G2_image3, keypoints_orb_left, keypoints_orb_right, G2_h13, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G2";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G2_image3, keypoints_orb_right, G2_path + keyType + "img3_key.png");
//		RunExactDetection(G2_image1, G2_image4, keypoints_orb_left, keypoints_orb_right, G2_h14, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G2";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G2_image4, keypoints_orb_right, G2_path + keyType + "img4_key.png");
//		RunExactDetection(G2_image1, G2_image5, keypoints_orb_left, keypoints_orb_right, G2_h15, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G2";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G2_image5, keypoints_orb_right, G2_path + keyType + "img5_key.png");
//		RunExactDetection(G2_image1, G2_image6, keypoints_orb_left, keypoints_orb_right, G2_h16, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G2";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G2_image6, keypoints_orb_right, G2_path + keyType + "img6_key.png");
//		//test G3
//		cout << "G3_start........" << endl;
//		RunExactDetection(G3_image1, G3_image2, keypoints_orb_left, keypoints_orb_right, G3_h12, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G3";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G3_image1, keypoints_orb_left, G3_path + keyType + "img1_key.png");
//		DrawSaveKeypoints(G3_image2, keypoints_orb_right, G3_path + keyType + "img2_key.png");
//		RunExactDetection(G3_image1, G3_image3, keypoints_orb_left, keypoints_orb_right, G3_h13, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G3";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G3_image3, keypoints_orb_right, G3_path + keyType + "img3_key.png");
//		RunExactDetection(G3_image1, G3_image4, keypoints_orb_left, keypoints_orb_right, G3_h14, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G3";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G3_image4, keypoints_orb_right, G3_path + keyType + "img4_key.png");
//		RunExactDetection(G3_image1, G3_image5, keypoints_orb_left, keypoints_orb_right, G3_h15, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G3";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G3_image5, keypoints_orb_right, G3_path + keyType + "img5_key.png");
//		RunExactDetection(G3_image1, G3_image6, keypoints_orb_left, keypoints_orb_right, G3_h16, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G3";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G3_image6, keypoints_orb_right, G3_path + keyType + "img6_key.png");
//       //test G4
//		cout << "G4_start........" << endl;
//		RunExactDetection(G4_image1, G4_image2, keypoints_orb_left, keypoints_orb_right, G4_h12, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G4";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G4_image1, keypoints_orb_left, G4_path + keyType + "img1_key.png");
//		DrawSaveKeypoints(G4_image2, keypoints_orb_right, G4_path + keyType + "img2_key.png");
//		RunExactDetection(G4_image1, G4_image3, keypoints_orb_left, keypoints_orb_right, G4_h13, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G4";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G4_image3, keypoints_orb_right, G4_path + keyType + "img3_key.png");
//		RunExactDetection(G4_image1, G4_image4, keypoints_orb_left, keypoints_orb_right, G4_h14, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G4";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G4_image4, keypoints_orb_right, G4_path + keyType + "img4_key.png");
//		RunExactDetection(G4_image1, G4_image5, keypoints_orb_left, keypoints_orb_right, G4_h15, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G4";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G4_image5, keypoints_orb_right, G4_path + keyType + "img5_key.png");
//		RunExactDetection(G4_image1, G4_image6, keypoints_orb_left, keypoints_orb_right, G4_h16, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G4";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G4_image6, keypoints_orb_right, G4_path + keyType + "img6_key.png");
//		//test G5
//		cout << "G5_start........" << endl;
//		RunExactDetection(G5_image1, G5_image2, keypoints_orb_left, keypoints_orb_right, G5_h12, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G5";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G5_image1, keypoints_orb_left, G5_path + keyType + "img1_key.png");
//		DrawSaveKeypoints(G5_image2, keypoints_orb_right, G5_path + keyType + "img2_key.png");
//		RunExactDetection(G5_image1, G5_image3, keypoints_orb_left, keypoints_orb_right, G5_h13, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G5";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G5_image3, keypoints_orb_right, G5_path + keyType + "img3_key.png");
//		RunExactDetection(G5_image1, G5_image4, keypoints_orb_left, keypoints_orb_right, G5_h14, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G5";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G5_image4, keypoints_orb_right, G5_path + keyType + "img4_key.png");
//		RunExactDetection(G5_image1, G5_image5, keypoints_orb_left, keypoints_orb_right, G5_h15, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G5";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G5_image5, keypoints_orb_right, G5_path + keyType + "img5_key.png");
//		RunExactDetection(G5_image1, G5_image6, keypoints_orb_left, keypoints_orb_right, G5_h16, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G5";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G5_image6, keypoints_orb_right, G5_path + keyType + "img6_key.png");
//		//test G6
//		cout << "G6_start........" << endl;
//		RunExactDetection(G6_image1, G6_image2, keypoints_orb_left, keypoints_orb_right, G6_h12, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G6";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G6_image1, keypoints_orb_left, G6_path + keyType + "img1_key.png");
//		DrawSaveKeypoints(G6_image2, keypoints_orb_right, G6_path + keyType + "img2_key.png");
//		RunExactDetection(G6_image1, G6_image3, keypoints_orb_left, keypoints_orb_right, G6_h13, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G6";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G6_image3, keypoints_orb_right, G6_path + keyType + "img3_key.png");
//		RunExactDetection(G6_image1, G6_image4, keypoints_orb_left, keypoints_orb_right, G6_h14, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G6";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G6_image4, keypoints_orb_right, G6_path + keyType + "img4_key.png");
//		RunExactDetection(G6_image1, G6_image5, keypoints_orb_left, keypoints_orb_right, G6_h15, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G6";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G6_image5, keypoints_orb_right, G6_path + keyType + "img5_key.png");
//		RunExactDetection(G6_image1, G6_image6, keypoints_orb_left, keypoints_orb_right, G6_h16, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G6";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G6_image6, keypoints_orb_right, G6_path + keyType + "img6_key.png");
//		//test G7
//		cout << "G7_start........" << endl;
//		RunExactDetection(G7_image1, G7_image2, keypoints_orb_left, keypoints_orb_right, G7_h12, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G7";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G7_image1, keypoints_orb_left, G7_path + keyType + "img1_key.png");
//		DrawSaveKeypoints(G7_image2, keypoints_orb_right, G7_path + keyType + "img2_key.png");
//		RunExactDetection(G7_image1, G7_image3, keypoints_orb_left, keypoints_orb_right, G7_h13, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G7";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G7_image3, keypoints_orb_right, G7_path + keyType + "img3_key.png");
//		RunExactDetection(G7_image1, G7_image4, keypoints_orb_left, keypoints_orb_right, G7_h14, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G7";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G7_image4, keypoints_orb_right, G7_path + keyType + "img4_key.png");
//		RunExactDetection(G7_image1, G7_image5, keypoints_orb_left, keypoints_orb_right, G7_h15, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G7";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G7_image5, keypoints_orb_right, G7_path + keyType + "img5_key.png");
//		RunExactDetection(G7_image1, G7_image6, keypoints_orb_left, keypoints_orb_right, G7_h16, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G7";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G7_image6, keypoints_orb_right, G7_path + keyType + "img6_key.png");
//		//test G8
//		cout << "G8_start........" << endl;
//		RunExactDetection(G8_image1, G8_image2, keypoints_orb_left, keypoints_orb_right, G8_h12, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G8";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G8_image1, keypoints_orb_left, G8_path + keyType + "img1_key.png");
//		DrawSaveKeypoints(G8_image2, keypoints_orb_right, G8_path + keyType + "img2_key.png");
//		RunExactDetection(G8_image1, G8_image3, keypoints_orb_left, keypoints_orb_right, G8_h13, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G8";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G8_image3, keypoints_orb_right, G8_path + keyType + "img3_key.png");
//		RunExactDetection(G8_image1, G8_image4, keypoints_orb_left, keypoints_orb_right, G8_h14, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G8";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G8_image4, keypoints_orb_right, G8_path + keyType + "img4_key.png");
//		RunExactDetection(G8_image1, G8_image5, keypoints_orb_left, keypoints_orb_right, G8_h15, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G8";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G8_image5, keypoints_orb_right, G8_path + keyType + "img5_key.png");
//		RunExactDetection(G8_image1, G8_image6, keypoints_orb_left, keypoints_orb_right, G8_h16, reapetality, nCrosCount, pORBexactor);
//		tempInformation.groupName = "G8";
//		tempInformation.keypoints_left = keypoints_orb_left.size();
//		tempInformation.keypoints_right = keypoints_orb_right.size();
//		tempInformation.nCrosCount = nCrosCount;
//		tempInformation.repeatelity = reapetality;
//		vecInoformation.push_back(tempInformation);
//		DrawSaveKeypoints(G8_image6, keypoints_orb_right, G8_path + keyType + "img6_key.png");
//		//
//	}
//	std::vector<InformationReapet> vecURsift;
//	//test ur sift
//	{
//		string keytype = "UR";
//		string txttype = "Scale";
//		//test G1;
//		{
//			std::vector<cv::KeyPoint> keypointsL, keypointsR;
//			float repeatl;
//			int ncroscount;
//			InformationReapet tempinfot;
//			cv::Ptr<FeatureDetector> ptt;
//			std::string strFeatureL = G1_path +"img1"+ txttype+".txt";
//			std::string strFeatureR = G1_path + "img2" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureL, keypointsL);
//			DrawSaveKeypoints(G1_image1, keypointsL, G1_path + keytype + "img1_key.png");
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G1_image2, keypointsR, G1_path + keytype + "img2_key.png");
//			evaluateFeatureDetector(G1_image1, G1_image2, G1_h12, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G1";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//			strFeatureR = G1_path + "img3" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G1_image3, keypointsR, G1_path + keytype + "img3_key.png");
//			evaluateFeatureDetector(G1_image1, G1_image3, G1_h13, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G1";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//			strFeatureR = G1_path + "img4" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G1_image4, keypointsR, G1_path + keytype + "img4_key.png");
//			evaluateFeatureDetector(G1_image1, G1_image4, G1_h14, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G1";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//			strFeatureR = G1_path + "img5" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G1_image5, keypointsR, G1_path + keytype + "img5_key.png");
//			evaluateFeatureDetector(G1_image1, G1_image5, G1_h15, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G1";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//			strFeatureR = G1_path + "img6" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G1_image6, keypointsR, G1_path + keytype + "img6_key.png");
//			evaluateFeatureDetector(G1_image1, G1_image6, G1_h16, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G1";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//		}
//		//test G2;
//		{
//			std::vector<cv::KeyPoint> keypointsL, keypointsR;
//			float repeatl;
//			int ncroscount;
//			InformationReapet tempinfot;
//			cv::Ptr<FeatureDetector> ptt;
//			std::string strFeatureL = G2_path + "img1"+txttype +".txt";
//			std::string strFeatureR = G2_path + "img2" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureL, keypointsL);
//			DrawSaveKeypoints(G2_image1, keypointsL, G2_path + keytype + "img1_key.png");
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G2_image2, keypointsR, G2_path + keytype + "img2_key.png");
//			evaluateFeatureDetector(G2_image1, G2_image2, G2_h12, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G2";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//			strFeatureR = G2_path + "img3" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G2_image3, keypointsR, G2_path + keytype + "img3_key.png");
//			evaluateFeatureDetector(G2_image1, G2_image3, G2_h13, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G2";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//			strFeatureR = G2_path + "img4" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G2_image4, keypointsR, G2_path + keytype + "img4_key.png");
//			evaluateFeatureDetector(G2_image1, G2_image4, G2_h14, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G2";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//			strFeatureR = G2_path + "img5" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G2_image5, keypointsR, G2_path + keytype + "img5_key.png");
//			evaluateFeatureDetector(G2_image1, G2_image5, G2_h15, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G2";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//			strFeatureR = G2_path + "img6" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G2_image6, keypointsR, G2_path + keytype + "img6_key.png");
//			evaluateFeatureDetector(G2_image1, G2_image6, G2_h16, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G2";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//		}
//		//test G3;
//		{
//			std::vector<cv::KeyPoint> keypointsL, keypointsR;
//			float repeatl;
//			int ncroscount;
//			InformationReapet tempinfot;
//			cv::Ptr<FeatureDetector> ptt;
//			std::string strFeatureL = G3_path + "img1" + txttype + ".txt";
//			std::string strFeatureR = G3_path + "img2" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureL, keypointsL);
//			DrawSaveKeypoints(G3_image1, keypointsL, G3_path + keytype + "img1_key.png");
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G3_image2, keypointsR, G3_path + keytype + "img2_key.png");
//			evaluateFeatureDetector(G3_image1, G3_image2, G3_h12, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G3";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//			strFeatureR = G3_path + "img3" + txttype + ".txt";
//			
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G3_image3, keypointsR, G3_path + keytype + "img3_key.png");
//			evaluateFeatureDetector(G3_image1, G3_image3, G3_h13, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G3";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//			strFeatureR = G3_path + "img4" + txttype + ".txt";
//			
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G3_image4, keypointsR, G3_path + keytype + "img4_key.png");
//			evaluateFeatureDetector(G3_image1, G3_image4, G3_h14, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G3";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//			strFeatureR = G3_path + "img5" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G3_image5, keypointsR, G3_path + keytype + "img5_key.png");
//			evaluateFeatureDetector(G3_image1, G3_image5, G3_h15, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G3";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//			strFeatureR = G3_path + "img6" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G3_image6, keypointsR, G3_path + keytype + "img6_key.png");
//			evaluateFeatureDetector(G3_image1, G3_image6, G3_h16, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G3";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//		}
//		//test G4;
//		{
//			std::vector<cv::KeyPoint> keypointsL, keypointsR;
//			float repeatl;
//			int ncroscount;
//			InformationReapet tempinfot;
//			cv::Ptr<FeatureDetector> ptt;
//			std::string strFeatureL = G4_path + "img1" + txttype + ".txt";
//			std::string strFeatureR = G4_path + "img2" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureL, keypointsL);
//			DrawSaveKeypoints(G4_image1, keypointsL, G4_path + keytype + "img1_key.png");
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G4_image2, keypointsR, G4_path + keytype + "img2_key.png");
//			evaluateFeatureDetector(G4_image1, G4_image2, G4_h12, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G4";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//			strFeatureR = G4_path + "img3" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G4_image3, keypointsR, G4_path + keytype + "img3_key.png");
//			evaluateFeatureDetector(G4_image1, G4_image3, G4_h13, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G4";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//			strFeatureR = G4_path + "img4" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G4_image4, keypointsR, G4_path + keytype + "img4_key.png");
//			evaluateFeatureDetector(G4_image1, G4_image4, G4_h14, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G4";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//			strFeatureR = G4_path + "img5" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G4_image5, keypointsR, G4_path + keytype + "img5_key.png");
//			evaluateFeatureDetector(G4_image1, G4_image5, G4_h15, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G4";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//			strFeatureR = G4_path + "img6" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G4_image6, keypointsR, G4_path + keytype + "img6_key.png");
//			evaluateFeatureDetector(G4_image1, G4_image6, G4_h16, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G4";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//		}
//		//test G5;
//		{
//			std::vector<cv::KeyPoint> keypointsL, keypointsR;
//			float repeatl;
//			int ncroscount;
//			InformationReapet tempinfot;
//			cv::Ptr<FeatureDetector> ptt;
//			std::string strFeatureL = G5_path + "img1" + txttype + ".txt";
//			std::string strFeatureR = G5_path + "img2" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureL, keypointsL);
//			DrawSaveKeypoints(G5_image1, keypointsL, G5_path + keytype + "img1_key.png");
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G5_image2, keypointsR, G5_path + keytype + "img2_key.png");
//			evaluateFeatureDetector(G5_image1, G5_image2, G5_h12, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G5";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//			strFeatureR = G5_path + "img3" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G5_image3, keypointsR, G5_path + keytype + "img3_key.png");
//			evaluateFeatureDetector(G5_image1, G5_image3, G5_h13, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G5";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//			strFeatureR = G5_path + "img4" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G5_image4, keypointsR, G5_path + keytype + "img4_key.png");
//			evaluateFeatureDetector(G5_image1, G5_image4, G5_h14, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G5";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//			strFeatureR = G5_path + "img5" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G5_image5, keypointsR, G5_path + keytype + "img5_key.png");
//			evaluateFeatureDetector(G5_image1, G5_image5, G5_h15, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G5";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//			strFeatureR = G5_path + "img6" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G5_image6, keypointsR, G5_path + keytype + "img6_key.png");
//			evaluateFeatureDetector(G5_image1, G5_image6, G5_h16, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G5";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//		}
//		//test G6;
//		{
//			std::vector<cv::KeyPoint> keypointsL, keypointsR;
//			float repeatl;
//			int ncroscount;
//			InformationReapet tempinfot;
//			cv::Ptr<FeatureDetector> ptt;
//			std::string strFeatureL = G6_path + "img1" + txttype + ".txt";
//			std::string strFeatureR = G6_path + "img2" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureL, keypointsL);
//			DrawSaveKeypoints(G6_image1, keypointsL, G6_path + keytype + "img1_key.png");
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G6_image2, keypointsR, G6_path + keytype + "img2_key.png");
//			evaluateFeatureDetector(G6_image1, G6_image2, G6_h12, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G6";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//			strFeatureR = G6_path + "img3" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G6_image3, keypointsR, G6_path + keytype + "img3_key.png");
//			evaluateFeatureDetector(G6_image1, G6_image3, G6_h13, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G6";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//			strFeatureR = G6_path + "img4" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G6_image4, keypointsR, G6_path + keytype + "img4_key.png");
//			evaluateFeatureDetector(G6_image1, G6_image4, G6_h14, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G6";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//			strFeatureR = G6_path + "img5" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G6_image5, keypointsR, G6_path + keytype + "img5_key.png");
//			evaluateFeatureDetector(G6_image1, G6_image5, G6_h15, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G6";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//			strFeatureR = G6_path + "img6" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G6_image6, keypointsR, G6_path + keytype + "img6_key.png");
//			evaluateFeatureDetector(G6_image1, G6_image6, G6_h16, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G6";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//		}
//		//test G7;
//		{
//			std::vector<cv::KeyPoint> keypointsL, keypointsR;
//			float repeatl;
//			int ncroscount;
//			InformationReapet tempinfot;
//			cv::Ptr<FeatureDetector> ptt;
//			std::string strFeatureL = G7_path + "img1" + txttype + ".txt";
//			std::string strFeatureR = G7_path + "img2" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureL, keypointsL);
//			DrawSaveKeypoints(G7_image1, keypointsL, G7_path + keytype + "img1_key.png");
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G7_image2, keypointsR, G7_path + keytype + "img2_key.png");
//			evaluateFeatureDetector(G7_image1, G7_image2, G7_h12, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G7";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//			strFeatureR = G7_path + "img3" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G7_image3, keypointsR, G7_path + keytype + "img3_key.png");
//			evaluateFeatureDetector(G7_image1, G7_image3, G7_h13, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G7";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//			strFeatureR = G7_path + "img4" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G7_image4, keypointsR, G7_path + keytype + "img4_key.png");
//			evaluateFeatureDetector(G7_image1, G7_image4, G7_h14, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G7";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//			strFeatureR = G7_path + "img5" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G7_image5, keypointsR, G7_path + keytype + "img5_key.png");
//			evaluateFeatureDetector(G7_image1, G7_image5, G7_h15, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G7";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//			strFeatureR = G7_path + "img6" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G7_image6, keypointsR, G7_path + keytype + "img6_key.png");
//			evaluateFeatureDetector(G7_image1, G7_image6, G7_h16, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G7";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//		}
//		//test G8;
//		{
//			std::vector<cv::KeyPoint> keypointsL, keypointsR;
//			float repeatl;
//			int ncroscount;
//			InformationReapet tempinfot;
//			cv::Ptr<FeatureDetector> ptt;
//			std::string strFeatureL = G8_path + "img1" + txttype + ".txt";
//			std::string strFeatureR = G8_path + "img2" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureL, keypointsL);
//			DrawSaveKeypoints(G8_image1, keypointsL, G8_path + keytype + "img1_key.png");
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G8_image2, keypointsR, G8_path + keytype + "img2_key.png");
//			evaluateFeatureDetector(G8_image1, G8_image2, G8_h12, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G8";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//			strFeatureR = G8_path + "img3" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G8_image3, keypointsR, G8_path + keytype + "img3_key.png");
//			evaluateFeatureDetector(G8_image1, G8_image3, G8_h13, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G8";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//			strFeatureR = G8_path + "img4" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G8_image4, keypointsR, G8_path + keytype + "img4_key.png");
//			evaluateFeatureDetector(G8_image1, G8_image4, G8_h14, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G8";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//			strFeatureR = G8_path + "img5" + txttype +".txt";
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G8_image5, keypointsR, G8_path + keytype + "img5_key.png");
//			evaluateFeatureDetector(G8_image1, G8_image5, G8_h15, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G8";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//			strFeatureR = G8_path + "img6" + txttype + ".txt";
//			ReadFeaturePoints(strFeatureR, keypointsR);
//			DrawSaveKeypoints(G8_image6, keypointsR, G8_path + keytype + "img6_key.png");
//			evaluateFeatureDetector(G8_image1, G8_image6, G1_h16, &keypointsL, &keypointsR, repeatl, ncroscount, ptt);
//			tempinfot.groupName = "G8";
//			tempinfot.keypoints_left = keypointsL.size();
//			tempinfot.keypoints_right = keypointsR.size();
//			tempinfot.nCrosCount = ncroscount;
//			tempinfot.repeatelity = repeatl;
//			vecURsift.push_back(tempinfot);
//		}
//   }
//	fstream fileOutrr;
//	fileOutrr.open(resultPath, std::ios::out);
//	for (int i=0;i<vecInoformation.size();++i)
//	{
//		fileOutrr << vecInoformation[i].groupName << "," << vecInoformation[i].keypoints_left << ","
//			<< vecInoformation[i].keypoints_right  << ","
//			<< vecInoformation[i].nCrosCount << "," << vecInoformation[i].repeatelity << endl;
//	}
//	fileOutrr.close();
//	fstream fileursift;
//	fileursift.open(resultURsiftPath,std::ios::out);
//	for (int i=0;i<vecURsift.size();++i)
//	{
//		fileursift << vecURsift[i].groupName << "," << vecURsift[i].keypoints_left << ","
//			<< vecURsift[i].keypoints_right << "," << vecURsift[i].nCrosCount << ","
//			<< vecURsift[i].repeatelity << endl;
//	}
//	fileursift.close();
//	float reapeat = 0;
//	int nCrossCount = 0;
//	//评估
//	/*string strFeature1 ="D:\\3DOpenSource\\featurepointsdata\\graf-viewpoint\\ppm\\img1.txt" ;
//	string strFeature2 ="D:\\3DOpenSource\\featurepointsdata\\graf-viewpoint\\ppm\\img2.txt" ;
//	ReadFeaturePoints(strFeature1, keypoints7);
//	ReadFeaturePoints(strFeature2, keypoints8);
//	evaluateFeatureDetector(image1, image2, G1_h12, &keypoints7, &keypoints8, reapeat, nCrossCount, pexactor5);
//	RunExactDetection(image1, image2, keypoints5, keypoints6,G1_h12, reapeat, nCrossCount, pexactor3);
//	std::cout << "reapeat=" <<reapeat<< endl;
//	std::cout << "ncrosscount=" << nCrossCount << endl;*/
//	//测试输出
//	//float repeat2 = 0;
//	//int ncurront2=0;
//	//RunExactDetection(image1, image2, keypoints7, keypoints8, HomoGraphy, repeat2, ncurront2, pexactor4);
//	//std::cout << "RANSAC:" << endl;
//	//std::cout << "reapeat2=" << repeat2 << endl;
//	//std::cout << "ncrosscount2=" << ncurront2 << endl;
//	//RunExactDetection(image1, image2, keypoints7, keypoints8, HomoGraphy1, repeat2, ncurront2, pexactor4);
//	//std::cout << "LMEDS:" << endl;
//	//std::cout << "reapeat2=" << repeat2 << endl;
//	//std::cout << "ncrosscount2=" << ncurront2 << endl;
//	//RunExactDetection(image1, image2, keypoints7, keypoints8, HomoGraphy2, repeat2, ncurront2, pexactor4);
//	//std::cout << "RHO:" << endl;
//	//std::cout << "reapeat2=" << repeat2 << endl;
//	//std::cout << "ncrosscount2=" << ncurront2 << endl;
//	//RunExactDetection(image1, image2, keypoints7, keypoints8, HomoGraphy3, repeat2, ncurront2, pexactor4);
//	//std::cout << "RANSAC_LMEDS:" << endl;
//	//std::cout << "reapeat2=" << repeat2 << endl;
//	//std::cout << "ncrosscount2=" << ncurront2 << endl;
//
//
//	//std::cout << "reapeat2=" << repeat2 << endl;
//	//std::cout << "ncrosscount2=" << ncurront2 << endl;
//	//string funde1 = "D://data//south-building//images//P118014";
//	//string funde2 = ".JPG";
//	//std::vector<string> strFind;
//	//FindImagelist(funde1, 1, 9, funde2, strFind);
//	//std::vector < std::vector<DMatch> > matchesf;
//	//std::vector<std::vector<uchar>> vecMatch;
//	//RunGetMatches(strFind, matchesf,vecMatch);
//	//std::vector<Point2f> vecPoints;
//	//RunRecallPrecisionCurve(matchesf,vecMatch, vecPoints);
//
//	//// Draw the epipolar lines
//	//std::vector<cv::Vec3f> lines1;
//	//cv::computeCorrespondEpilines(points1, 1, fundamental, lines1);
//
//	//for (std::vector<cv::Vec3f>::const_iterator it = lines1.begin();
//	//	it != lines1.end(); ++it) {
//
//	//	cv::line(image2, cv::Point(0, -(*it)[2] / (*it)[1]),
//	//		cv::Point(image2.cols, -((*it)[2] + (*it)[0] * image2.cols) / (*it)[1]),
//	//		cv::Scalar(255, 255, 255));
//	//}
//
//	//std::vector<cv::Vec3f> lines2;
//	//cv::computeCorrespondEpilines(points2, 2, fundamental, lines2);
//
//	//for (std::vector<cv::Vec3f>::const_iterator it = lines2.begin();
//	//	it != lines2.end(); ++it) {
//
//	//	cv::line(image1, cv::Point(0, -(*it)[2] / (*it)[1]),
//	//		cv::Point(image1.cols, -((*it)[2] + (*it)[0] * image1.cols) / (*it)[1]),
//	//		cv::Scalar(255, 255, 255));
//	//}
//
//	// Display the images with epipolar lines
//	//cv::imshow("Right Image Epilines (RANSAC)", image1);
//	//cv::imshow("Left Image Epilines (RANSAC)", image2);
//	//cv::imwrite("D:\\data\\south-building\\images\\contactRansacEPL.JPG", image1);
//	//cv::imwrite("D:\\data\\south-building\\images\\contactRansacEPR.JPG", image2);
//
//	cv::waitKey(0);
//	return 0;
//}