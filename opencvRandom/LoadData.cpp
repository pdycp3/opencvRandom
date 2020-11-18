#include "LoadData.h"
void readDescriptor(std::vector<std::vector<float>> &dataset, std::vector<int> &labels, const char* dataPath)
{
	FILE* dataFile = fopen(dataPath, "rb");
	if (dataFile == NULL)
	{
		printf("打开%s失败", dataPath);
		return;
	}

	int nRow, nCol;
	fread(&nRow, sizeof(int), 1, dataFile);
	fread(&nCol, sizeof(int), 1, dataFile);
	printf("nRow=%d", nRow);
	printf("nCol=%d", nCol);

	//float temp;
	int temp;//影像id
	dataset.reserve(nRow);
	labels.reserve(nRow);
	float* ftemp1 = new float[128];//特征描述符
	for (int i = 0; i < nRow; i++)
	{
		std::vector<float> vecdata;
		fread(&temp, sizeof(int), 1, dataFile);
		fread(ftemp1, sizeof(float), 128, dataFile);
		labels.push_back(temp);
		for (int j = 0; j < 128; ++j)
			vecdata.push_back( ftemp1[j]);
		dataset.push_back(vecdata);
	}
	fclose(dataFile);
	delete[] ftemp1;
}

void convertTwoDimensionArarayToMat(float ** dataset, cv::Mat & traindata)
{
	int nRow = traindata.rows;
	int nCol = traindata.cols;
	for (int i=0;i<nRow;++i)
	{
		for (int j=0;j<nCol;++j)
		{
			traindata = dataset[i][j];

		}
	}

}

int CharToNum(char cInput)
{
	int nReturn = cInput - 'A';
	return nReturn;

}

//String::String(const char * str)
//{
//	if (str==NULL)
//	{
//		printf("please input valid string\n");
//		this->m_data = NULL;
//		return;
//	}
//	this->m_data = new char[strlen(str) + 1];
//	strcpy(this->m_data, str);
//}
//
//String::String(const String & other)
//{
//	if (other.m_data == NULL)
//	{
//		printf("please input valid string\n");
//		this->m_data = NULL;
//		return;
//	}
//	this->m_data = new char[strlen(other.m_data) + 1];
//	strcpy(this->m_data, other.m_data);
//}
//
//String::~String(void)
//{
//	if (this->m_data)
//	{
//		delete this->m_data;
//		this->m_data = NULL;
//	}
//
//}
//
//String & String::operator=(const String & other)
//{
//	if (this != &other)
//	{
//		delete[] this->m_data;
//		if (!other.m_data)
//		{
//			this->m_data = 0;
//		}
//		else
//		{
//			this->m_data = new char[strlen(other.m_data) + 1];
//			strcpy(this->m_data, other.m_data);
//		}
//	}
//	return *this;
//
//
//}
