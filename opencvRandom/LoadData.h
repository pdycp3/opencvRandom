#pragma once
#ifndef LOADDATA_H
#define LOADDATA_H

#include <opencv2/core/core.hpp>
//读取特征描述符数据
void readDescriptor(std::vector<std::vector<float>> &dataset,std::vector<int> &labels, const char* dataPath);
void convertTwoDimensionArarayToMat(float **dataset,cv::Mat & traindata);
int  CharToNum(char cInput);

#endif

