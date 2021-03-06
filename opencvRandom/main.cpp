﻿
#include "RandomTree.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/ml/ml.hpp"  
#include "LoadData.h"
#include <fstream>
#include <iostream>
#include <strstream>
#define TRAIN_NUM 746604
#ifndef STD_API
#define STD_API __stdcall
#endif
#define TEST_NUM 208860
#define FEATURE 128
#define NUMBER_OF_CLASSES 20
#define NUMSELECT 4
#define NUMTREES 100
#define DEPTHTREE 10
#define MINNUM 10
#define NUMIMAGE 55
using namespace std;
using namespace cv;
using namespace ml;
template <typename T>
vector<size_t> sort_indexes_e(vector<T> &v)
{
	vector<size_t> idx(v.size());
	for (size_t i = 0; i < v.size(); ++i)
		idx[i] = i;
	sort(idx.begin(), idx.end(),
		[&v](size_t i1, size_t i2) {return v[i1] > v[i2]; });
	return idx;
}
cv::Ptr<cv::ml::TrainData> prepare_train_data(const cv::Mat& data, const cv::Mat& responses, int ntrain_samples)
{
	using namespace cv;
	Mat sample_idx = Mat::zeros(1, data.rows, CV_8U);
	Mat train_samples = sample_idx.colRange(0, ntrain_samples);
	train_samples.setTo(Scalar::all(1));

	int nvars = data.cols;
	Mat var_type(nvars + 1, 1, CV_8U);
	var_type.setTo(Scalar::all(ml::VAR_ORDERED));
	var_type.at<uchar>(nvars) = ml::VAR_CATEGORICAL;

	return ml::TrainData::create(data, ml::ROW_SAMPLE, responses, noArray(), sample_idx, noArray(), noArray());
}
/*template <typename T1,typename T2>*/
bool UniqueInsert(map<int,double> &mapInsert,int a,double b)
{
	if (!mapInsert.empty())
	{
		auto itert = mapInsert.begin();
		for (auto x:mapInsert)
		{
			if (x.first==a)
				break;
			itert++;
		}
		if (itert!=mapInsert.end())
		{
			if (b<mapInsert[a])
			{
				mapInsert[a] = b;
				return true;
			}
			else
			{
				return true;
			}
			
		}
		else
		{
			mapInsert.insert(pair<int, double>(a, b));
			return true;
		}


	}
	else
	{
		mapInsert.insert(pair<int, double>(a, b));
		return true;
	}
	return false;

}

class  CProcessBase
{
public:
	/**
	* @brief 构造函数
	*/
	CProcessBase()
	{
		m_dPosition = 0.0;
		m_iStepCount = 100;
		m_iCurStep = 0;
		m_bIsContinue = true;
	}

	/**
	* @brief 析构函数
	*/
	virtual ~CProcessBase() {}

	/**
	* @brief 设置进度信息
	* @param pszMsg			进度信息
	*/
	virtual void SetMessage(const char* pszMsg) = 0;

	/**
	* @brief 设置进度值
	* @param dPosition		进度值
	* @return 返回是否取消的状态，true为不取消，false为取消
	*/
	virtual bool SetPosition(double dPosition) = 0;

	/**
	* @brief 进度条前进一步，返回true表示继续，false表示取消
	* @return 返回是否取消的状态，true为不取消，false为取消
	*/
	virtual bool StepIt() = 0;

	/**
	* @brief 设置进度个数
	* @param iStepCount		进度个数
	*/
	virtual void SetStepCount(int iStepCount)
	{
		ReSetProcess();
		m_iStepCount = iStepCount;
	}

	/**
	* @brief 获取进度信息
	* @return 返回当前进度信息
	*/
	string GetMessage()
	{
		return m_strMessage;
	}

	/**
	* @brief 获取进度值
	* @return 返回当前进度值
	*/
	double GetPosition()
	{
		return m_dPosition;
	}

	/**
	* @brief 重置进度条
	*/
	void ReSetProcess()
	{
		m_dPosition = 0.0;
		m_iStepCount = 100;
		m_iCurStep = 0;
		m_bIsContinue = true;
	}

	/*! 进度信息 */
	string m_strMessage;
	/*! 进度值 */
	double m_dPosition;
	/*! 进度个数 */
	int m_iStepCount;
	/*! 进度当前个数 */
	int m_iCurStep;
	/*! 是否取消，值为false时表示计算取消 */
	bool m_bIsContinue;
};
class CConsoleProcess : public CProcessBase
{
public:
	/**
	* @brief 构造函数
	*/
	CConsoleProcess()
	{
		m_dPosition = 0.0;
		m_iStepCount = 100;
		m_iCurStep = 0;
	};

	/**
	* @brief 析构函数
	*/
	~CConsoleProcess()
	{
		//remove(m_pszFile);
	};

	/**
	* @brief 设置进度信息
	* @param pszMsg			进度信息
	*/
	void SetMessage(const char* pszMsg)
	{
		m_strMessage = pszMsg;
		printf("%s\n", pszMsg);
	}

	/**
	* @brief 设置进度值
	* @param dPosition		进度值
	* @return 返回是否取消的状态，true为不取消，false为取消
	*/
	bool SetPosition(double dPosition)
	{
		m_dPosition = dPosition;
		TermProgress(m_dPosition);
		m_bIsContinue = true;
		return true;
	}

	/**
	* @brief 进度条前进一步
	* @return 返回是否取消的状态，true为不取消，false为取消
	*/
	bool StepIt()
	{
		m_iCurStep++;
		m_dPosition = m_iCurStep * 1.0 / m_iStepCount;

		TermProgress(m_dPosition);
		m_bIsContinue = true;
		return true;
	}

private:
	void TermProgress(double dfComplete)
	{
		static int nLastTick = -1;
		int nThisTick = (int)(dfComplete * 40.0);

		nThisTick = MIN(40, MAX(0, nThisTick));

		// Have we started a new progress run?  
		if (nThisTick < nLastTick && nLastTick >= 39)
			nLastTick = -1;

		if (nThisTick <= nLastTick)
			return;

		while (nThisTick > nLastTick)
		{
			nLastTick++;
			if (nLastTick % 4 == 0)
				fprintf(stdout, "%d", (nLastTick / 4) * 10);
			else
				fprintf(stdout, ".");
		}

		if (nThisTick == 40)
			fprintf(stdout, " - done.\n");
		else
			fflush(stdout);
	}
};

struct ImageSave 
{
	int nQueryId;
	int nQueryLabel;
	int nDstLabel;
	int nDstId;
	float dMatchDist;
};
struct SingleImageSave 
{
	int nQueryId;
	float  dMatchDist;
};


static bool cmpImasave(const ImageSave & a, const ImageSave & b)
{
	return a.dMatchDist < b.dMatchDist;
}
static bool cmpImasaveId(const ImageSave & a, const ImageSave & b)
{
	return a.nQueryId < b.nQueryId;
}
static bool cmpSISDist(const SingleImageSave &a, const SingleImageSave & b)
{
	return a.dMatchDist < b.dMatchDist;
}
class RandomKDTreeMatch 
{
public:
	struct RKDPARAM
	{
		int nTrees;
		int nPerTreeSave;
		int nTotalSave;
		//knn参数
		int nSearchParam;
	};
	struct SearchResult
	{
		int id;
		float distance;
	};
	static bool cmpl(const SearchResult & a, const SearchResult & b)
	{
		return a.distance < b.distance;
	}
	RandomKDTreeMatch() {};
	~RandomKDTreeMatch() {
		m_vecRandomKdtrees.clear();
		m_vecQueryVector.clear();
		m_vecMatSource.clear();
	};
	void InitialParam(RKDPARAM & rParam);
	void SetSourceData(std::vector<std::vector<float>> & vecSourceData, CProcessBase * pProcess = NULL);
	void CreatKDTrees(CProcessBase * pProcess);
	void SetQueryData(std::vector<std::vector<float>> & vecQueryVector, CProcessBase * pProcess);
	void QuerySearch(std::vector<std::vector<SearchResult>> & queryResult, CProcessBase * pProcess);
	void SetSourceData1(std::vector<std::vector<float>> & vecSourceData, CProcessBase * pProcess = NULL);
	void CreatKDTrees1(CProcessBase * pProcess);
	void QuerySearch1(std::vector<std::vector<SearchResult>> & queryResult, CProcessBase * pProcess);
private:
	//存储kd树的容器
	std::vector<cv::flann::Index *> m_vecRandomKdtrees;
	//查询向量
	std::vector<std::vector<float>> m_vecQueryVector;
	//构建索引树的特征向量
	std::vector<cv::Mat> m_vecMatSource;
	std::vector<vector<int>> m_vecSourceIndex;
	//方法1️的成员变量
	cv::Mat m_MatSource1;
	cv::flann::Index * m_RDTrees;
	RKDPARAM m_RKDParam;
};
void RandomKDTreeMatch::InitialParam(RKDPARAM & rParam)
{
	m_RKDParam.nPerTreeSave = rParam.nPerTreeSave;
	m_RKDParam.nSearchParam = rParam.nSearchParam;
	m_RKDParam.nTotalSave = rParam.nTotalSave;
	m_RKDParam.nTrees = rParam.nTrees;
}
void RandomKDTreeMatch::SetSourceData(std::vector<std::vector<float>> & vecSourceData,CProcessBase * pProcess)
{
	int nSize = vecSourceData.size();
	if (nSize <= 0)
	{
		cout << "please check your train dataset" << endl;
		return;
	}
	RNG &rng = theRNG();
	int nTree = m_RKDParam.nTrees > 0 ? m_RKDParam.nTrees : 20;
	if (pProcess!=NULL)
	{
		pProcess->ReSetProcess();
		pProcess->SetMessage("start set dataset");
	}
	if ((!m_vecMatSource.empty())||(!m_vecSourceIndex.empty()))
	{
		m_vecMatSource.clear();
		m_vecSourceIndex.clear();
	}
	int nCol = vecSourceData[0].size();
	for (int l=0;l<nTree;++l)
	{
		if (pProcess!=NULL)
		{
			pProcess->SetStepCount(nTree);
		}
		cv::Mat initialmat;
		
		std::vector<int> vec_index;
		vec_index.reserve(nSize);
		
		for (int i = 0; i < nSize; ++i)
		{
			int j = rng.uniform(0, nSize);
			vec_index.push_back(j);
			cv::Mat Sample(1, nCol, cv::DataType<float>::type, vecSourceData[j].data());
			initialmat.push_back(Sample);
			
		}
		pProcess->SetPosition((l + 1.0) / nTree);
		m_vecSourceIndex.push_back(vec_index);
		m_vecMatSource.push_back(initialmat);
	}
	if (pProcess!=NULL)
	{
		pProcess->SetMessage("finish set dataset");
	}
}
void RandomKDTreeMatch::SetSourceData1(std::vector<std::vector<float>> & vecSourceData, CProcessBase * pProcess)
{
	int nSize = vecSourceData.size();
	if (nSize <= 0)
	{
		cout << "please check your train dataset" << endl;
		return;
	}
	int nTree = m_RKDParam.nTrees > 0 ? m_RKDParam.nTrees : 20;
	if (pProcess != NULL)
	{
		pProcess->ReSetProcess();
		pProcess->SetMessage("start set dataset");
	}
	if (!m_MatSource1.empty())
	{
		m_MatSource1.release();
	}
	int nCol = vecSourceData[0].size();
	cv::Mat initialmat;
	for (int i = 0; i < nSize; ++i)
	{
		cv::Mat Sample(1, nCol, cv::DataType<float>::type, vecSourceData[i].data());
		m_MatSource1.push_back(Sample);
	}
	if (pProcess != NULL)
	{
		pProcess->SetMessage("finish set dataset");
	}
}
void RandomKDTreeMatch::CreatKDTrees(CProcessBase * pProcess)
{
	int nTree = m_RKDParam.nTrees > 0 ? m_RKDParam.nTrees : 20;
	int nSize = m_vecMatSource.size();
	if (nTree!=nSize)
	{
		cout << "please check source data and param";
		return;
	}
	if (pProcess!=NULL)
	{
		pProcess->ReSetProcess();
		pProcess->SetMessage("start creat trees");
		pProcess->SetStepCount(nTree);
	}
	for (int i=0;i<nTree;++i)
	{
		cv::flann::KDTreeIndexParams indexparam(1);
		m_vecMatSource[i].convertTo(m_vecMatSource[i], CV_32F);
		cv::flann::Index * pKdTree = new cv::flann::Index(m_vecMatSource[i],indexparam);
		m_vecRandomKdtrees.push_back(pKdTree);
		pProcess->SetPosition((i + 1)*1.0 / nTree);
	}
	if (pProcess!=NULL)
	{
		pProcess->SetMessage("creat trees was finished");
	}
}
void RandomKDTreeMatch::CreatKDTrees1(CProcessBase * pProcess)
{
	int nTree = m_RKDParam.nTrees > 0 ? m_RKDParam.nTrees : 20;
	if (pProcess != NULL)
	{
		pProcess->ReSetProcess();
		pProcess->SetMessage("start creat trees");
	}
	cv::flann::KDTreeIndexParams indexparam(nTree);
	if (m_RDTrees!=NULL)
	{
		m_RDTrees->release();
	}
	m_RDTrees= new cv::flann::Index(m_MatSource1, indexparam);

	if (pProcess != NULL)
	{
		pProcess->SetMessage("creat trees was finished");
	}
}
void RandomKDTreeMatch::SetQueryData(std::vector<std::vector<float>> & vecQueryVector, CProcessBase * pProcess)
{
	if (!m_vecQueryVector.empty())
	{
		m_vecQueryVector.clear();
	}
	if (pProcess != NULL)
	{
		pProcess->SetMessage("start query data set");
	}
	m_vecQueryVector = vecQueryVector;
	if (pProcess!=NULL)
	{
		pProcess->SetMessage("finish query data set");
	}
}
void RandomKDTreeMatch::QuerySearch(std::vector<std::vector<SearchResult>> & queryResult,CProcessBase * pProcess)
{
	if (m_vecQueryVector.size()<=0)
	{
		cout << "please set query vector first" << endl;
	}
	int nSize = m_vecQueryVector.size();
	queryResult.reserve(nSize);
	cv::flann::SearchParams params(m_RKDParam.nSearchParam);
	if (pProcess!=NULL)
	{
		pProcess->ReSetProcess();
		pProcess->SetMessage("start query............");
		pProcess->SetStepCount(nSize);
	}
	for (int i=0;i<nSize;++i)
	{
		std::vector<SearchResult> vecQueryResult;
		SearchResult searchResult;
		for (int j=0;j<m_vecRandomKdtrees.size();++j)
		{
			vector<float> queryDist(m_RKDParam.nPerTreeSave);
			vector<int> vecIndex(m_RKDParam.nPerTreeSave);
			
			m_vecRandomKdtrees[j]->knnSearch(m_vecQueryVector[i], vecIndex, queryDist, m_RKDParam.nPerTreeSave, params);
			for (int k=0;k<m_RKDParam.nPerTreeSave;++k)
			{
				searchResult.id = m_vecSourceIndex[j][vecIndex[k]];
				searchResult.distance = queryDist[k];
				vecQueryResult.push_back(searchResult);
			}
		}
		queryResult.push_back(vecQueryResult);
		pProcess->SetPosition((i + 1)*1.0 / nSize);
	}
	if (pProcess!=NULL)
	{
		pProcess->SetMessage("query finished..............");
	}


}
void RandomKDTreeMatch::QuerySearch1(std::vector<std::vector<SearchResult>> & queryResult, CProcessBase * pProcess)
{
	if (m_vecQueryVector.size() <= 0)
	{
		cout << "please set query vector first" << endl;
	}
	int nSize = m_vecQueryVector.size();
	queryResult.reserve(nSize);
	cv::flann::SearchParams params(m_RKDParam.nSearchParam);
	if (pProcess != NULL)
	{
		pProcess->ReSetProcess();
		pProcess->SetMessage("start query............");
		pProcess->SetStepCount(nSize);
	}
	for (int i = 0; i < nSize; ++i)
	{
		std::vector<SearchResult> vecQueryResult;
		SearchResult searchResult;
		vector<float> queryDist(m_RKDParam.nPerTreeSave);
		vector<int> vecIndex(m_RKDParam.nPerTreeSave);
		m_RDTrees->knnSearch(m_vecQueryVector[i], vecIndex, queryDist, m_RKDParam.nPerTreeSave, params);
		for (int k = 0; k < m_RKDParam.nPerTreeSave; ++k)
		{
			searchResult.id = vecIndex[k];
			searchResult.distance = queryDist[k];
			vecQueryResult.push_back(searchResult);
		}
		queryResult.push_back(vecQueryResult);
		pProcess->SetPosition((i + 1)*1.0 / nSize);
	}
	if (pProcess != NULL)
	{
		pProcess->SetMessage("query finished..............");
	}
}
int main(int argc, char* argv[])
{	

	vector<vector<float>>trainset;
	vector<int> trainlabels;
	//输入数据
	//string strFeatrueBinary= "G:\\data\\arialimage20\\uav20.ibx";
	//string strSaveRtree = "G:\\data\\arialimage20\\uav20RtreeModel.xml";
	//string strSaveResult = "G:\\data\\arialimage20\\uav20result.txt";
	//string strHeatResult = "G:\\data\\arialimage20\\uav20heatScaleLn.txt";
	//string strFeatrueBinary = "G:\\data\\south-building\\southbuilding.ibx";
	//string strSaveRtree = "G:\\data\\south-building\\southbuilding.xml";
	//string strSaveResult = "G:\\data\\south-building\\southbuilding.txt";
	//string strHeatResult = "G:\\data\\south-building\\southbuildingHeat.txt";
	//string strFeatrueBinary = "G:\\data\\uavtest\\uavexhaust.ibx";
	//string strSaveRtree = "G:\\data\\uavtest\\uavexhaustRtreeModel.xml";
	//string strSaveResult = "G:\\data\\uavtest\\uavexhaustresult.xml";
	//string strFeatrueBinary = "D:\\3DOpenSource\\Cameratest\\camertest.ibx";
	//string strSaveRtree = "D:\\3DOpenSource\\Cameratest\\camertest.xml";
	//string strSaveResult = "D:\\3DOpenSource\\Cameratest\\camertest.txt";
	//string strHeatResult = "D:\\3DOpenSource\\Cameratest\\camertestheat.txt";
	string strFeatrueBinary = "G:\\data\\unorderedtest\\1\\1.ibx";
	string strSaveRtree = "G:\\data\\unorderedtest\\1\\1.xml";
	string strSaveResult = "G:\\data\\unorderedtest\\1\\1test.txt";
	string strHeatResult = "G:\\data\\unorderedtest\\1\\1heat.txt";

	 //RTrees for classification


	//加载描述子
	readDescriptor(trainset, trainlabels, strFeatrueBinary.c_str());
	//test yuanyangben shuju 
	//cv::Mat initialmat;
	//int nSize = trainset.size();
	//std::vector<int> vec_index;
	//std::vector<float> vecDist;
	//vecDist.reserve(2);
	//vec_index.reserve(2);

	//for (int i = 0; i < nSize; ++i)
	//{
	//	cv::Mat Sample(1, 128, cv::DataType<float>::type, trainset[i].data());
	//	initialmat.push_back(Sample);

	//}
	//int nCooount = 0;
	//int nCCount = 0;
	//cv::flann::KDTreeIndexParams indexparam(10);
	//cv::flann::Index* ptree = new cv::flann::Index(initialmat, indexparam);
	//for (int i=0;i<nSize;++i)
	//{
	//	cv::Mat Sample(1, 128, cv::DataType<float>::type, trainset[i].data());
	//	ptree->knnSearch(Sample, vec_index, vecDist, 2);
	//	if (vec_index[0]==i)
	//	{
	//		nCooount++;
	//	}
	//	if (vecDist[1]<1500)
	//	{
	//		nCCount++;
	//	}


	//}
	//cout << "preceision=" << nCooount * 1.0 / nSize << endl;
	//cout << "dist=" << nCCount * 1.0 / nSize << endl;



	//建立随机kdtree
	std::vector<std::vector<RandomKDTreeMatch::SearchResult>> vecResult;
	RandomKDTreeMatch *pRandomKDTreeMatch = new RandomKDTreeMatch();
	RandomKDTreeMatch::RKDPARAM rparam;
	rparam.nTrees = 5;
	rparam.nPerTreeSave = 2;
	rparam.nSearchParam = 32;
	rparam.nTotalSave = 5;
	pRandomKDTreeMatch->InitialParam(rparam);
	CProcessBase* pProcess = new CConsoleProcess();
	pRandomKDTreeMatch->SetSourceData(trainset, pProcess);
	pRandomKDTreeMatch->CreatKDTrees(pProcess);
	pRandomKDTreeMatch->SetQueryData(trainset, pProcess);
	pRandomKDTreeMatch->QuerySearch(vecResult, pProcess);
	vector<vector<float>> vecDistanceCount;
	vector<vector<int>> vecNumCount;
	vector<vector<float>> vecOverScore;
	//初始化存储中间变量空间
	for (int i=0;i<NUMIMAGE;++i)
	{
		vector<float> vectmep(NUMIMAGE, 0);
		vector<int> vecNum(NUMIMAGE, 0);
		vector<float> vecoverlap(NUMIMAGE, 0);
		vecDistanceCount.push_back(vectmep);
		vecNumCount.push_back(vecNum);
		vecOverScore.push_back(vecoverlap);
	}
	float nTotalSize = vecResult.size()*1.0/100;
	int nSize = vecResult.size();
	//进度定义
	int j = 10;
	//存储每张影像的匹配点对
	std::vector<vector<ImageSave>> vecAllImagesave;
	vecAllImagesave.reserve(NUMIMAGE);
	int nLastLabel = 0;
	//统计当前影像的存储内容
	vector<ImageSave> vecImage;
	int nLabelIdLast = trainlabels[0];
	for (size_t i = 0; i < nSize; i++)
	{
		//输出当前特征描述符的查询结果
		vector<RandomKDTreeMatch::SearchResult> &vectemp=vecResult[i];
	    //当前点的查询结果的距离从小到大进行排序
		sort(vectemp.begin(), vectemp.end(), RandomKDTreeMatch::cmpl);
		//初始化相同匹配点对id
		int nNextId = vecResult[i][0].id;
		int nSamplesave = 0;
		
		for (int k=0;k<vecResult[i].size();++k)
		{
			int nCurrenid = vectemp[k].id;
			//移除同一个点互匹配
			if (nCurrenid ==i)
				continue;
			//移除相同匹配点对
			if (nCurrenid ==nNextId)
	            continue;
			nNextId = nCurrenid;
			//移除同一张影像的匹配
			if (trainlabels[i]==trainlabels[nCurrenid])
				continue;
			int nSourecelabel = trainlabels[i];
			int nCurrenlabel = trainlabels[nCurrenid];
			//开始存储，查询向量由原向量组成，那么每张影像的邻接向量查询全部存储下来
			if (trainlabels[i]!=nLastLabel)
			{
				vecAllImagesave.push_back(vecImage);
				vecImage.clear();
				nLastLabel = trainlabels[i];
			}

			ImageSave stuImagesave;
			stuImagesave.dMatchDist = vectemp[k].distance;
			stuImagesave.nQueryId = i;
			stuImagesave.nQueryLabel = nSourecelabel;
			stuImagesave.nDstId = vectemp[k].id;
			stuImagesave.nDstLabel = nCurrenlabel;
			vecImage.push_back(stuImagesave);
			
			//开始统计
			//匹配点数统计
		
			//vecNumCount[nSourecelabel][nCurrenlabel] += 1;
			//距离统计由于采用的是欧式距离的平方为了防止数过大采用平均值来进行统计

			//if (vecNumCount[nSourecelabel][nCurrenlabel]==1)
			//{
			//	vecDistanceCount[nSourecelabel][nCurrenlabel] += vectemp[k].distance;
			//
			//}
			//else {
			//	vecDistanceCount[nSourecelabel][nCurrenlabel] += (vectemp[k].distance - vecDistanceCount[nSourecelabel][nCurrenlabel]) / vecNumCount[nSourecelabel][nCurrenlabel];
			//
			//}
			
		}
		float dProcess = i / nTotalSize;
	    if (dProcess>j)
	    {
			printf(">>>%f%%", dProcess);
			j += 10;
	    }
	}
 	vecAllImagesave.push_back(vecImage);
	printf("\n");
     //筛选当前距离最近的查询点对
	//for (int i=0;i<NUMIMAGE;++i)
	//{
	//	int nSaveSize = vecAllImagesave[i].size();
	//	vector<ImageSave> &vecISve = vecAllImagesave[i];
	//	sort(vecISve.begin(), vecISve.end(), cmpImasave);
	//	float dMaxdist = vecISve[nSaveSize-1].dMatchDist;
	//	float dMindisat = vecISve[0].dMatchDist;
	//	float d2Min = dMaxdist * 0.1;
	//	for (int j=0;j<nSaveSize;++j)
	//	{
	//		if (vecISve[j].dMatchDist<=d2Min)
	//		{
	//			int nQL = vecISve[j].nQueryLabel;
	//			int nDL = vecISve[j].nDstLabel;
	//			vecNumCount[nQL][nDL] += 1;
	//			if (vecNumCount[nQL][nDL]==1)
	//			{
	//				vecDistanceCount[nQL][nDL] += vecISve[j].dMatchDist;
	//			}
	//			else
	//			{
	//				vecDistanceCount[nQL][nDL] += (vecISve[j].dMatchDist - vecDistanceCount[nQL][nDL]) / vecNumCount[nQL][nDL];
	//			}
	//		}
	//	}
	//}
	//版本二
	std::vector<std::vector<ImageSave>> vecCount;
	for (int i = 0; i < NUMIMAGE*NUMIMAGE; ++i)
	{
		vector<ImageSave> vectemp;
		vecCount.push_back(vectemp);
	}
	for (int i = 0; i < NUMIMAGE; ++i)
	{
		int nSaveSize = vecAllImagesave[i].size();
		vector<ImageSave> &vecISve = vecAllImagesave[i];
		sort(vecISve.begin(), vecISve.end(), cmpImasave);
		float dMaxdist = vecISve[nSaveSize - 1].dMatchDist;
		float dMindisat = vecISve[0].dMatchDist;
		float d2Min = dMaxdist * 0.1;
		//float d2Min = dMindisat * 3;
		for (int j = 0; j < nSaveSize; ++j)
		{
			if (vecISve[j].dMatchDist <= d2Min)
			{
				int nQL = vecISve[j].nQueryLabel;
				int nDL = vecISve[j].nDstLabel;
				vecNumCount[nQL][nDL] += 1;
				int nId = nDL + nQL * NUMIMAGE;
				vecCount[nId].push_back(vecISve[j]);
			}
		}
	}
	//按照查询id排序,并移除相同id
	for (int i=0;i<NUMIMAGE;++i)
	{
		for (int j=0;j<NUMIMAGE;++j)
		{
			int nCheck = j + i * NUMIMAGE;
			if (vecCount[nCheck].size()<=1)
				continue;
			sort(vecCount[nCheck].begin(), vecCount[nCheck].end(), cmpImasaveId);
			map<int, double> mapContainer;
			for (int j=0;j<vecCount[nCheck].size();++j)
			{
				int nQuid = vecCount[nCheck][j].nQueryId;
				double nMdist = vecCount[nCheck][j].dMatchDist;
				bool bInsertState = UniqueInsert(mapContainer, nQuid, nMdist);
			}
			for (auto t:mapContainer)
			{
				vecNumCount[i][j] += 1;
				if (vecNumCount[i][j] == 1)
				{
					vecDistanceCount[i][j] += t.second;
				}
				else
				{
					vecDistanceCount[i][j] += (t.second - vecDistanceCount[i][j]) / vecNumCount[i][j];
				}
			}
		}
	}
	//版本三
	//for (int i = 0; i < NUMIMAGE; ++i)
	//{
	//	for (int j = 0; j < NUMIMAGE; ++j)
	//	{
	//		int nCheck = j + i * NUMIMAGE;
	//		if (vecCount[nCheck].size() <= 1)
	//			continue;
	//		sort(vecCount[nCheck].begin(), vecCount[nCheck].end(), cmpImasaveId);
	//		map<int, double> mapContainer, mapConvert;
	//		for (int j = 0; j < vecCount[nCheck].size(); ++j)
	//		{
	//			int nQuid = vecCount[nCheck][j].nQueryId;
	//			double nMdist = vecCount[nCheck][j].dMatchDist;
	//			bool bInsertState = UniqueInsert(mapContainer, nQuid, nMdist);
	//		}
	//		//构早匹配距离点对移除距离较大的点对
	//		vector<SingleImageSave> vecImageSaveOne;
	//		for (auto y : mapContainer)
	//		{
	//			SingleImageSave tempsingleiamge;
	//			tempsingleiamge.nQueryId = y.first;
	//			tempsingleiamge.dMatchDist = y.second;
	//			vecImageSaveOne.push_back(tempsingleiamge);
	//		}
	//		std::sort(vecImageSaveOne.begin(), vecImageSaveOne.end(), cmpSISDist);
	//		int nSizeSave = vecImageSaveOne.size()*0.3;
	//		for (int k = 0; k < nSizeSave; ++k)
	//		{
	//			mapConvert.insert(pair<int, double>(vecImageSaveOne[k].nQueryId, vecImageSaveOne[k].dMatchDist));
	//		}


	//		//移除同一幅影像的错误点对引入ransac
	//		std::cout << "mapsize=" << mapConvert.size() << endl;
	//		for (auto t : mapConvert)
	//		{
	//			vecNumCount[i][j] += 1;
	//			if (vecNumCount[i][j] == 1)
	//			{
	//				vecDistanceCount[i][j] += t.second;
	//			}
	//			else
	//			{
	//				vecDistanceCount[i][j] += (t.second - vecDistanceCount[i][j]) / vecNumCount[i][j];
	//			}
	//		}
	//	}
	//}

	//不同版本的输出为个数和距离的统计
	//开始评分统计
	for (int i=0;i<NUMIMAGE;++i)
	{
		for (int j=0;j<NUMIMAGE;++j)
		{
			if(i==j)
				continue;
			if (vecNumCount[i][j]<=30)
				continue;
			float dSqrt = sqrtf(vecDistanceCount[i][j]);
			float dScore = log(vecNumCount[i][j] * exp(-dSqrt / vecNumCount[i][j]));
			vecOverScore[i][j] = dScore<=0?0:dScore;
		}
	}
	//输出到文件
	//for (int i=0;i<NUMIMAGE-1;++i)
	//{
	//	for (int j=i+1;j<NUMIMAGE;++j)
	//	{
	//		double dMean = (vecOverScore[i][j] + vecOverScore[j][i]) / 2;
	//		vecOverScore[i][j] = vecOverScore[j][i] = dMean;
	//	}
	//}
	//for (int i=0;i<NUMIMAGE;++i)
	//{
	//	vector<size_t>  vecIndex;
	//	vecIndex = sort_indexes_e(vecOverScore[i]);
	//	for (int j=4;j<vecIndex.size();++j)
	//	{
	//		vecOverScore[i][vecIndex[j]] = 0;
	//	}
	//}
	ofstream fileout;
	fileout.open(strHeatResult.c_str(), std::ios::out);
	for (int i=0;i<NUMIMAGE;++i)
	{
		for (int j=0;j<NUMIMAGE;++j)
			fileout << vecOverScore[i][j] << ",";
		fileout << endl;
	}
	fileout.close();
	return 0;
}

