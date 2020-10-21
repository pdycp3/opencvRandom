//
////#include <opencv2/core/core.hpp>
////#include <opencv2/highgui/highgui.hpp>
////#include <opencv2/features2d/features2d.hpp>
////#include "opencv2/imgproc/imgproc.hpp"  
////#include "opencv2/ml/ml.hpp"  
//#include "LoadData.h"
////
////#include <iostream>
////#include <strstream>
//#define TRAIN_NUM 66075
////#define TRAIN_NUM 100
//#define TEST_NUM 10000
//#define FEATURE 128
//#define NUMBER_OF_CLASSES 10
//#define NUMSELECT 4
//#define NUMTREES 100
//#define DEPTHTREE 10
//#define MINNUM 10
//#define NUMIMAGE 5
////using namespace std;
////using namespace cv;
////
//////bool read_num_class_data(const string& fileFeatureTrain, int numF, int fLen, cv::Mat* _data, cv::Mat* _responses)
//////{
//////	using namespace cv;
//////	Mat el_ptr(1, numF, CV_32F);
//////	vector<int>  responses(0);
//////	_data->release();
//////	_responses->release();
//////
//////	freopen(fileFeatureTrain.c_str(), "r", stdin);
//////	cout << "The feature is loading....." << endl;
//////
//////	int i = 0;
//////	int label = 0;
//////	for (int i = 0; i < numF; ++i) {
//////		StyleFeature  aFeat; 
//////		aFeat.second.resize(fLen);
//////		std::string sline; getline(cin, sline);
//////
//////		//�Կո�ֿ�
//////		int idxBlank = sline.find_first_of(" ");
//////
//////		std::string sLabel = sline;//��ȡ��ǩ��
//////		sLabel.erase(idxBlank, sLabel.length());
//////		responses.push_back(label);//aFeat.first = label = atoi(sLabel.c_str());
//////
//////		std::string sFV = sline;
//////		sFV.erase(0, idxBlank + 1);//��ȡһ�У�����
//////
//////		int idxFv = 0;
//////		float fV = 0.0;
//////		while (sFV.length() > 0 && idxFv < fLen) {
//////			int idxColon = sFV.find_first_of(":");
//////			std::string sv = sFV;
//////			std::strstream ssv;
//////			sv = sv.substr(idxColon + 1, sv.find_first_of(" ") - 2);
//////			ssv << sv; ssv >> fV;
//////			el_ptr.at<float>(i) = fV;//aFeat.second[idxFv] = fV;
//////
//////			++idxFv;
//////			sFV.erase(0, sFV.find_first_of(" ") + 1);
//////		}
//////		_data->push_back(el_ptr);//trainData.push_back(aFeat);
//////	}
//////
//////	fclose(stdin); cout << "The feature load over....." << endl;
//////	Mat(responses).copyTo(*_responses);
//////
//////	return true;
//////}
////cv::Ptr<cv::ml::TrainData> prepare_train_data(const cv::Mat& data, const cv::Mat& responses, int ntrain_samples)
////{
////	using namespace cv;
////	Mat sample_idx = Mat::zeros(1, data.rows, CV_8U);
////	Mat train_samples = sample_idx.colRange(0, ntrain_samples);
////	train_samples.setTo(Scalar::all(1));
////
////	int nvars = data.cols;
////	Mat var_type(nvars + 1, 1, CV_8U);
////	var_type.setTo(Scalar::all(ml::VAR_ORDERED));
////	var_type.at<uchar>(nvars) = ml::VAR_CATEGORICAL;
////
////	return ml::TrainData::create(data, ml::ROW_SAMPLE, responses, noArray(), sample_idx, noArray(), noArray());
////}
////
////int main(int argc, char* argv[])
////{
////	//if (argc < 9) {
////	//	std::cout << "argc<9";
////	//	return 0;
////	//}
////	//����������������ʱ����
////	//std::string fileFeatureTrain(argv[1]);
////	//std::string  fileFeatureTest(argv[2]);
////	//std::string        fileTrees(argv[3]);
////
////	//int lenF = atoi(argv[4]);//�������� 32
////	//int numF = atoi(argv[5]);//ʹ���������� 1000
////	//int nsample = atoi(argv[6]);//�������� ����numF
////	//int nTrees = atoi(argv[7]);
////	//int nClass = atoi(argv[8]);
////	//1. prepare data
////	float**trainset;
////	float** testset;
////	float*trainlabels;
////	float*testlabels;
////
////	//ѵ�����ݼ�60000��float����
////	//��֤�������ݼ���10000����float����
////	//ÿһ��������784ά
////
////	trainset = new float*[TRAIN_NUM];
////	testset = new float*[TEST_NUM];
////	trainlabels = new float[TRAIN_NUM];
////	testlabels = new float[TEST_NUM];
////	for (int i = 0; i < TRAIN_NUM; ++i)
////	{
////		trainset[i] = new float[FEATURE];
////	}
////	for (int i = 0; i < TEST_NUM; ++i)
////	{
////		testset[i] = new float[FEATURE];
////	}
////	//��������
////	string strFeatrueBinary= "D:\\data\\colmatest\\features.ibx";
////	string strSaveRtree = "D:\\data\\colmatest\\RtreeModel.xml";
////	// RTrees for classification
////	//��������
////	Mat data;
////	Mat responses;
////	//����������
////	readDescriptor(trainset, trainlabels, strFeatrueBinary.c_str());
////	//ת��
////	data =  Mat(TRAIN_NUM, 128, CV_32F, trainset);
////    responses = Mat( TRAIN_NUM,1, CV_32F, trainlabels);
////	//anothoer�汾
////	//Ptr<ml::TrainData> tdata = ml::TrainData::create(data, ml::ROW_SAMPLE, responses);
////	// ����������
////	Ptr<ml::RTrees> model;
////	model = ml::RTrees::create();
////	//�������������
////	model->setMaxDepth(10);
////	//�ڵ���С��������
////	model->setMinSampleCount(10);
////	//�ع�������ֹ��׼
////	model->setRegressionAccuracy(0);
////	//�Ƿ���������ѵ�
////	model->setUseSurrogates(false);
////	//���������
////	model->setMaxCategories(15);
////	//�������������
////	model->setPriors(Mat());
////	//����ı�����Ҫ��
////	model->setCalculateVarImportance(true);
////	//���ڵ����ѡ��������Ӽ��Ĵ�С
////	model->setActiveVarCount(4);
////	//��ֹ��׼
////	model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + (0.01f > 0 ? TermCriteria::EPS : 0), 100, 0.01f));
////	//ѵ��ģ��
////	model->train(data, ml::ROW_SAMPLE, responses);
////	//����ѵ����ɵ�ģ��
////	//model->save("filename_to_save.xml");
////
////	double train_hr = 0, test_hr = 0;
////	// ����ѵ���Ͳ������ݵ�Ԥ�����
////
////	//ѵ�����ݵ�Ԥ�����
////
////	//���ɭ���е�������
////	cout << "Number of trees: " << model->getRoots().size() << endl;
////
////
////
////	cout << "ending____________________________________________________" << endl;
////
////
////
////
////	return 0;
////}
//#include "opencv2/core/core.hpp"
//#include "opencv2/ml/ml.hpp"
//#include <cstdio>
//#include <vector>
//#include <iostream>
////#include <string>
//
//using namespace std;
//using namespace cv;
//using namespace cv::ml;
//
//
//char  * strcpy(char  * strDest, const   char  * strSrc)
//
//{
//	if ((strDest == NULL) || (strSrc == NULL))
//	{
//		printf("please input valid string");
//		return;
//	}
//	
//
//	char  * address = strDest;
//
//	while ((*strDest++ = *strSrc++) != '\0') {}
//	return   address;
//
//}
////void GetMemory(char *p)
////{
////	p = (char *)malloc(100);
////}
////void Test(void)
////{
////	char *str = NULL;
////	GetMemory(str);
////	strcpy(str, "hello world");
////	printf(str);
////}
////char *GetMemory(void)
////{
////	char p[] = "hello world";
////	return p;
////}
////void Test(void)
////{
////	char *str = NULL;
////	str = GetMemory();
////	printf(str);
////}
////void GetMemory2(char **p, int num)
////{
////	*p = (char *)malloc(num);
////}
////void Test(void)
////{
////	char *str = NULL;
////	GetMemory2(&str, 100);
////	strcpy(str, "hello");
////	printf(str);
////}
//void Test(void)
//{
//	char *str = (char *)malloc(100);
//	strcpy(str, "hello");
//	free(str);
//	if (str != NULL)
//	{
//		strcpy(str, "world");
//		printf(str);
//	}
//}
//
//// ���ļ��ж�ȡdata��responses 
//int read_num_class_data(const string& filename, int var_count, Mat* _data, Mat* _responses)
//{
//	const int M = 1024;
//	char buf[M + 2];
//	Mat el_ptr(1, var_count, CV_32F);
//	vector<int> responses;
//	_data->release();
//	_responses->release();
//	//fָ��洢���ݵĵ�ַ
//	FILE* f = fopen(filename.c_str(), "rt");
//	if (!f)
//	{
//		cout << "Could not read the database " << filename << endl;
//		return -1;
//	}
//
//	for (;;)
//	{
//		char* ptr;
//		int i;
//		//fgets���ļ��ж�ȡһ�����ݴ��뻺����
//		//strchr�����ַ���buf���״γ��֣���λ��
//		if (!fgets(buf, M, f) || !strchr(buf, ','))
//			break;
//		responses.push_back((int)buf[0]);
//		ptr = buf + 2;
//		for (i = 0; i < var_count; i++)
//		{
//			int n = 0;
//			//��ȡ��ʽ�����ַ����е�����
//			sscanf(ptr, "%f%n", &el_ptr.at<float>(i), &n);
//			ptr += n + 1;
//		}
//		if (i < var_count)
//			break;
//		_data->push_back(el_ptr);
//	}
//	fclose(f);
//
//	Mat(responses).copyTo(*_responses);
//	cout << "The database " << filename << " is loaded.\n";
//	return 0;
//}
//#include <iostream>
//#include<opencv2/opencv.hpp>
//
//using namespace cv;
//
///*
//@Param_input src :the source Image
//@Param_input bCounterClock: whether counter-clock
//@Param_output dst:the destination Image
//*/
//void RotateAngle(Mat &src, float angle, bool bCounterClock, Mat &dst)
//{
//	float dRotate = 0;
//	if (bCounterClock)
//	{
//		dRotate = (float)(angle / 180.0 * CV_PI);
//	}
//	else
//	{
//		dRotate = (float)(-angle / 180.0 * CV_PI);
//	}
//
//
//	//���ͼ��
//	int maxBorder = (int)(max(src.cols, src.rows)* 1.414); //��Ϊsqrt(2)*max
//	int dx = (maxBorder - src.cols) / 2;
//	int dy = (maxBorder - src.rows) / 2;
//	copyMakeBorder(src, dst, dy, dy, dx, dx, BORDER_CONSTANT);
//
//	//��ת
//	Point2f center((float)(dst.cols / 2), (float)(dst.rows / 2));
//	Mat affine_matrix = getRotationMatrix2D(center, dRotate, 1.0);//�����ת����
//	warpAffine(dst, dst, affine_matrix, dst.size());
//
//	//����ͼ����ת֮�����ͼ������ľ���
//	float sinVal = abs(sin(dRotate));
//	float cosVal = abs(cos(dRotate));
//	Size targetSize((int)(src.cols * cosVal + src.rows * sinVal),
//		(int)(src.cols * sinVal + src.rows * cosVal));
//
//	//��������߿�
//	int x = (dst.cols - targetSize.width) / 2;
//	int y = (dst.rows - targetSize.height) / 2;
//	Rect rect(x, y, targetSize.width, targetSize.height);
//	dst = Mat(dst, rect);
//}
//
//int main() {
//	cv::Mat src = cv::imread("testimage.bmp");
//	cv::Mat dst;
//	RotateAngle(src, 15, false, dst);
//	cv::imshow("src", src);
//	cv::imshow("dst", dst);
//	cv::waitKey(0);
//	return 0;
//}

//int main()
//{
//	//string data_filename = "D:\\data\\colmatest\\features.data";
//	//Mat data;
//	//Mat responses;
//	////��ȡdata��responses
//	//read_num_class_data(data_filename, 128, &data, &responses);
//
//	//int nsamples_all = data.rows;  //��������
//	//int ntrain_samples = (int)(nsamples_all*0.8);  //ѵ����������
//	//cout << "Training the classifier ...\n" << endl;
//	//Mat sample_idx = Mat::zeros(1, data.rows, CV_8U);
//	//int nvars = data.cols;
//	//Mat var_type(nvars + 1, 1, CV_8U);
//	//var_type.setTo(Scalar::all(VAR_ORDERED));
//	//var_type.at<uchar>(nvars) = VAR_CATEGORICAL;
//	////ѵ������
//	//Ptr<TrainData> tdata = TrainData::create(data, ROW_SAMPLE, responses, noArray(), sample_idx, noArray(), var_type);
//	////���id
//	//for (int i=0;i<10;++i)
//	//{
//	//	cout << sample_idx.at<int>(i)<<endl;
//	//}
//	//// ����������
//	//Ptr<RTrees> model;
//	//model = RTrees::create();
//	////�������������
//	//model->setMaxDepth(10);
//	////�ڵ���С��������
//	//model->setMinSampleCount(10);
//	////�ع�������ֹ��׼
//	//model->setRegressionAccuracy(0);
//	////�Ƿ���������ѵ�
//	//model->setUseSurrogates(false);
//	////���������
//	//model->setMaxCategories(15);
//	////�������������
//	//model->setPriors(Mat());
//	////����ı�����Ҫ��
//	//model->setCalculateVarImportance(true);
//	////���ڵ����ѡ��������Ӽ��Ĵ�С
//	//model->setActiveVarCount(4);
//	////��ֹ��׼
//	//model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + (0.01f > 0 ? TermCriteria::EPS : 0), 100, 0.01f));
//	////ѵ��ģ��
//	//model->train(tdata);
//	////����ѵ����ɵ�ģ��
//	////model->save("D:\\data\\colmatest\\filename_to_save.xml");
//
//	//double train_hr = 0, test_hr = 0;
//	//// ����ѵ���Ͳ������ݵ�Ԥ�����
//	//for (int i = 0; i < nsamples_all; i++)
//	//{
//	//	Mat sample = data.row(i);
//	//	float r = model->predict(sample);
//	//	//�ж�Ԥ���Ƿ���ȷ������ֵС����СֵFLT_EPSILON��
//	//	r = std::abs(r - responses.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;
//	//	//����
//	//	if (i < ntrain_samples)
//	//		train_hr += r;
//	//	else
//	//		test_hr += r;
//	//}
//	////ѵ�����ݵ�Ԥ�����
//	//test_hr /= nsamples_all - ntrain_samples;
//	////�������ݵ�Ԥ�����
//	//train_hr = ntrain_samples > 0 ? train_hr / ntrain_samples : 1.;
//	//printf("Recognition rate: train = %.1f%%, test = %.1f%%\n", train_hr*100., test_hr*100.);
//	////���ɭ���е�������
//	//cout << "Number of trees: " << model->getRoots().size() << endl;
//	//// ������Ҫ��
//	//Mat var_importance = model->getVarImportance();
//	//if (!var_importance.empty())
//	//{
//	//	double rt_imp_sum = sum(var_importance)[0];
//	//	printf("var#\timportance (in %%):\n");
//	//	int i, n = (int)var_importance.total();
//	//	for (i = 0; i < n; i++)
//	//		printf("%-2d\t%-4.1f\n", i, 100.f*var_importance.at<float>(i) / rt_imp_sum);
//	//}
//
//
//	system("pause");
//
//
//
//
//
//
//
//
//
//	return 0;
//
//
//}

