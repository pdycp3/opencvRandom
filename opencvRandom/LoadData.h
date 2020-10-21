#pragma once
#ifndef LOADDATA_H
#define LOADDATA_H

#include <opencv2/core/core.hpp>
//读取特征描述符数据
void readDescriptor(float** dataset, char*labels, const char* dataPath);
void convertTwoDimensionArarayToMat(float **dataset,cv::Mat & traindata);
int  CharToNum(char cInput);

//class String
//{
//public:
//	String(const char *str = NULL);	// 普通构造函数
//	String(const String &other);	    // 拷贝构造函数
//	~String(void);					    // 析构函数
//	String & operator = (const String & other);
//		// 赋值函数
//private:
//	char  	*m_data;				// 用于保存字符串
//};




#endif

