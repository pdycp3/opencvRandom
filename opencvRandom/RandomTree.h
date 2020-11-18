#pragma once
#ifndef RANDOM_TREE_H
#define RANDOM_TREE_H
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <string>
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

class RandomTree
{
public:
	RandomTree(string name);
	void set_train_data_(vector<vector<float>> &train_data, vector<int> &label);
	void Train();
	void Save();
	void GetTrainVotes(vector<vector<int>>& sampleVote);
	float AccOnTrain();
	int Predict(vector<float> &f);
private:
	string name_;
	Ptr<ml::TrainData> train_data_;
	Ptr<ml::RTrees> forest_;
};

RandomTree::RandomTree(string name)
{
	name_ = name;
	forest_ = cv::ml::RTrees::create();
	forest_->setMaxDepth(10); //树的最大深度
	forest_->setPriors(cv::Mat());
	forest_->setRegressionAccuracy(0.01); //设置回归精度
	//终止标准
	forest_->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 100, 0.01));
	forest_->setMinSampleCount(10); //节点的最小样本数量
	forest_->setUseSurrogates(false);
	forest_->setMaxCategories(15);
	forest_->setCalculateVarImportance(true); //计算变量的重要性
	forest_->setActiveVarCount(4);            //树节点随机选择特征子集的大小
}

void RandomTree::set_train_data_(vector<vector<float>> &train_data, vector<int> &label)
{
	int feat_num = train_data[0].size();
	// cout << feat_num << endl;
	cv::Mat train_mat;
	for (int i = 0; i < train_data.size(); ++i)
	{
		cv::Mat ele(1, feat_num, CV_32F);
		for (int j = 0; j < feat_num; ++j)
		{
			ele.at<float>(j) = train_data[i][j];
		}
		train_mat.push_back(ele);
	}
	train_data_ = ml::TrainData::create(train_mat, ml::ROW_SAMPLE, Mat(label));
	cout << "train data is over" << endl;
}

void RandomTree::Train()
{
	cout << "start train data" << endl;
	forest_->train(train_data_);
	cout << " train data over" << endl;
}

void RandomTree::Save()
{
	string path = "../model/" + name_ + ".xml";
	// cout << path << endl;
	forest_->save(path);
}

float RandomTree::AccOnTrain()
{
	cv::Mat train_sample = train_data_->getTrainSamples();
	cv::Mat train_label = train_data_->getTrainResponses();
	int size = train_sample.rows, cnt = 0;
	// cout << size << endl;
	for (int i = 0; i < size; ++i)
	{
		cv::Mat sample = train_sample.row(i);
		int r = forest_->predict(sample);
		// cout << train_label.at<int>(i) << endl;
		if (r == train_label.at<int>(i))
			cnt++;
	}
	return 1.0 * cnt / size;
}

int RandomTree::Predict(vector<float> &f)
{
	int feat_num = f.size();
	cv::Mat ele(1, feat_num, CV_32F);
	for (int j = 0; j < feat_num; ++j)
	{
		ele.at<float>(j) = f[j];
	}
	return forest_->predict(ele);
}
void RandomTree::GetTrainVotes(vector<vector<int>> &sampleVote)
{
	cout << "start get vote" << endl;
	cv::Mat train_sample = train_data_->getTrainSamples();
	cv::Mat train_label = train_data_->getTrainResponses();
	int size = train_sample.rows, cnt = 0;
	for (int i = 0; i < size; ++i)
	{
		vector<int> vecVotes;
		cv::Mat sample = train_sample.row(i);
		cv::Mat resulte;
		forest_->getVotes(sample,resulte,0);
		vecVotes.push_back(train_label.at<int>(i));
		const int * pLabel = resulte.ptr<int>(0);
		const int * pVote = resulte.ptr<int>(1);
		for (int j = 0; j < resulte.cols; ++j)
		{
			vecVotes.push_back(pLabel[j]);
			vecVotes.push_back(pVote[j]);
		}
			
		sampleVote.push_back(vecVotes);
	}
	cout << "get vote over" << endl;
}
#endif
