#pragma once
#ifndef LOADDATA_H
#define LOADDATA_H

#include <opencv2/core/core.hpp>
//��ȡ��������������
void readDescriptor(float** dataset, char*labels, const char* dataPath);
void convertTwoDimensionArarayToMat(float **dataset,cv::Mat & traindata);
int  CharToNum(char cInput);

//class String
//{
//public:
//	String(const char *str = NULL);	// ��ͨ���캯��
//	String(const String &other);	    // �������캯��
//	~String(void);					    // ��������
//	String & operator = (const String & other);
//		// ��ֵ����
//private:
//	char  	*m_data;				// ���ڱ����ַ���
//};




#endif

