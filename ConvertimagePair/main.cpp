#include<iostream>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <fstream>
#include <vector>
#include <string.h>
using namespace std;
void ConVertImageList(string strDeepImagePair,string strOutput)
{
	ifstream strInput;
	strInput.open(strDeepImagePair.c_str());
	ofstream strOuput;
	strOuput.open(strOutput.c_str(), std::ios::out);
	int NReadRow = 1;
	int nStartSplit = 2;
	int nRow = 0;
	int** pCount=nullptr;
	//获取第一行的影像的数量
	while (!strInput.eof())
	{
		string strcurrentLine;
		getline(strInput, strcurrentLine);
		if (NReadRow==1)
		{
			int nNumber = std::atoi(strcurrentLine.c_str());
			nStartSplit += nNumber;
			nRow = nNumber;
			pCount = new int*[nNumber];
			for (int i=0;i<nNumber;++i)
			{
				pCount[i] = new int[nNumber];
				for (int j=0;j<nNumber;++j)
				{
					pCount[i][j] = 0;
				}
			}
		}
		else
		{
			if (NReadRow>=nStartSplit)
			{
				vector<string > strList;
				boost::split(strList, strcurrentLine, boost::is_any_of(":"), boost::token_compress_on);
				if (strList.size()>1)
				{
					int nFirstNum = std::atoi(strList[0].c_str());
					vector<string> strList2;
					boost::split(strList2, strList[1], boost::is_any_of(" "), boost::token_compress_on);
					if (strList2.size()==1)
					{
						int nsecond = std::atoi(strList2[0].c_str());
						pCount[nFirstNum][nsecond] = 1;
						
					}
					else
					{
						

						for (int k = 0; k < strList2.size()-1; ++k)
						{
							int nSecontNum = std::atoi(strList2[k].c_str());
							pCount[nFirstNum][nSecontNum] = 1;
						}

					}
					
				}
				

			}




		}
		NReadRow++;


	}
	for (int n=0;n<nRow;++n)
	{
		for (int m=0;m<nRow;++m)
		{
			strOuput << pCount[n][m] << ",";
		}
		strOuput << endl;
	}
	
	
	strOuput.close();
	strInput.close();












}
int main()
{
	string str = "D:\\data\\ConvertTest\\image_pairs.txt";
	string out = "D:\\data\\ConvertTest\\image_pairsout.txt";
	ConVertImageList(str, out);


}