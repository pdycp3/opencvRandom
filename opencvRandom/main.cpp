
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
#define TRAIN_NUM 208860
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
#define NUMIMAGE 5
using namespace std;
using namespace cv;
using namespace ml;
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
class  CProcessBase
{
public:
	/**
	* @brief ���캯��
	*/
	CProcessBase()
	{
		m_dPosition = 0.0;
		m_iStepCount = 100;
		m_iCurStep = 0;
		m_bIsContinue = true;
	}

	/**
	* @brief ��������
	*/
	virtual ~CProcessBase() {}

	/**
	* @brief ���ý�����Ϣ
	* @param pszMsg			������Ϣ
	*/
	virtual void SetMessage(const char* pszMsg) = 0;

	/**
	* @brief ���ý���ֵ
	* @param dPosition		����ֵ
	* @return �����Ƿ�ȡ����״̬��trueΪ��ȡ����falseΪȡ��
	*/
	virtual bool SetPosition(double dPosition) = 0;

	/**
	* @brief ������ǰ��һ��������true��ʾ������false��ʾȡ��
	* @return �����Ƿ�ȡ����״̬��trueΪ��ȡ����falseΪȡ��
	*/
	virtual bool StepIt() = 0;

	/**
	* @brief ���ý��ȸ���
	* @param iStepCount		���ȸ���
	*/
	virtual void SetStepCount(int iStepCount)
	{
		ReSetProcess();
		m_iStepCount = iStepCount;
	}

	/**
	* @brief ��ȡ������Ϣ
	* @return ���ص�ǰ������Ϣ
	*/
	string GetMessage()
	{
		return m_strMessage;
	}

	/**
	* @brief ��ȡ����ֵ
	* @return ���ص�ǰ����ֵ
	*/
	double GetPosition()
	{
		return m_dPosition;
	}

	/**
	* @brief ���ý�����
	*/
	void ReSetProcess()
	{
		m_dPosition = 0.0;
		m_iStepCount = 100;
		m_iCurStep = 0;
		m_bIsContinue = true;
	}

	/*! ������Ϣ */
	string m_strMessage;
	/*! ����ֵ */
	double m_dPosition;
	/*! ���ȸ��� */
	int m_iStepCount;
	/*! ���ȵ�ǰ���� */
	int m_iCurStep;
	/*! �Ƿ�ȡ����ֵΪfalseʱ��ʾ����ȡ�� */
	bool m_bIsContinue;
};
class CConsoleProcess : public CProcessBase
{
public:
	/**
	* @brief ���캯��
	*/
	CConsoleProcess()
	{
		m_dPosition = 0.0;
		m_iStepCount = 100;
		m_iCurStep = 0;
	};

	/**
	* @brief ��������
	*/
	~CConsoleProcess()
	{
		//remove(m_pszFile);
	};

	/**
	* @brief ���ý�����Ϣ
	* @param pszMsg			������Ϣ
	*/
	void SetMessage(const char* pszMsg)
	{
		m_strMessage = pszMsg;
		printf("%s\n", pszMsg);
	}

	/**
	* @brief ���ý���ֵ
	* @param dPosition		����ֵ
	* @return �����Ƿ�ȡ����״̬��trueΪ��ȡ����falseΪȡ��
	*/
	bool SetPosition(double dPosition)
	{
		m_dPosition = dPosition;
		TermProgress(m_dPosition);
		m_bIsContinue = true;
		return true;
	}

	/**
	* @brief ������ǰ��һ��
	* @return �����Ƿ�ȡ����״̬��trueΪ��ȡ����falseΪȡ��
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

class RandomKDTreeMatch 
{
public:
	struct RKDPARAM
	{
		int nTrees;
		int nPerTreeSave;
		int nTotalSave;
		//knn����
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
private:
	//�洢kd��������
	std::vector<cv::flann::Index *> m_vecRandomKdtrees;
	//��ѯ����
	std::vector<std::vector<float>> m_vecQueryVector;
	//��������������������
	std::vector<cv::Mat> m_vecMatSource;
	std::vector<vector<int>> m_vecSourceIndex;
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
int main(int argc, char* argv[])
{	
	vector<vector<float>>trainset;
	vector<int> trainlabels;
	//��������
	string strFeatrueBinary= "G:\\data\\arialimage20\\uav20.ibx";
	string strSaveRtree = "G:\\data\\arialimage20\\uav20RtreeModel.xml";
	string strSaveResult = "G:\\data\\arialimage20\\uav20result.txt";
	//string strFeatrueBinary = "G:\\data\\uavtest\\uavexhaust.ibx";
	//string strSaveRtree = "G:\\data\\uavtest\\uavexhaustRtreeModel.xml";
	//string strSaveResult = "G:\\data\\uavtest\\uavexhaustresult.xml";
	 //RTrees for classification


	//����������
	readDescriptor(trainset, trainlabels, strFeatrueBinary.c_str());
	//�������kdtree
	std::vector<std::vector<RandomKDTreeMatch::SearchResult>> vecResult;
	RandomKDTreeMatch *pRandomKDTreeMatch = new RandomKDTreeMatch();
	RandomKDTreeMatch::RKDPARAM rparam;
	rparam.nTrees = 10;
	rparam.nPerTreeSave = 3;
	rparam.nSearchParam = 32;
	rparam.nTotalSave = 7;
	pRandomKDTreeMatch->InitialParam(rparam);
	CProcessBase* pProcess = new CConsoleProcess();
	pRandomKDTreeMatch->SetSourceData(trainset, pProcess);
	pRandomKDTreeMatch->CreatKDTrees(pProcess);
	pRandomKDTreeMatch->SetQueryData(trainset, pProcess);
	pRandomKDTreeMatch->QuerySearch(vecResult, pProcess);
	vector<vector<float>> vecDistanceCount(NUMIMAGE);
	vector<vector<int>> vecNumCount(NUMIMAGE);
	vector<vector<float>> vecOverScore(NUMIMAGE);
	for (int i=0;i<NUMIMAGE;++i)
	{
		vector<float> vectmep(NUMIMAGE, 0);
		vector<int> vecNum(NUMIMAGE, 0);
		vector<float> vecoverlap(NUMIMAGE, 0);
		vecDistanceCount.push_back(vectmep);
		vecNumCount.push_back(vecNum);
		vecOverScore.push_back(vecoverlap);
	}
	for (size_t i = 0; i < vecResult.size(); i++)
	{
		vector<float>  vecTemp = trainset[i];
		vector<RandomKDTreeMatch::SearchResult> &vectemp=vecResult[i];
		sort(vectemp.begin(), vectemp.end(), RandomKDTreeMatch::cmpl);
		
		
		for (int k=0;k<vecResult[i].size();++k)
		{
			
			if (vectemp[k].id==i)
				continue;
			
		    
			
		}
	
		cout << "test" << endl;

	}







	


	return 0;
}

