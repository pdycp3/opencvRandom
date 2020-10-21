//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/features2d/features2d.hpp>
//#include "opencv2/imgproc/imgproc.hpp"  
#include <io.h>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
using namespace std;

void getAllFiles(string path, vector<string>& files) {
	//文件句柄
	long hFile = 0;
	//文件信息
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1) {
		do {
			if ((fileinfo.attrib & _A_SUBDIR)) { //比较文件类型是否是文件夹
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0) {
					files.push_back(p.assign(path).append("\\").append(fileinfo.name));
					//递归搜索
					getAllFiles(p.assign(path).append("\\").append(fileinfo.name), files);
				}
			}
			else {
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0); //寻找下一个，成功返回0，否则-1
		_findclose(hFile);
	}
}
int main()
{
	//check get all files
	vector<string> currentPathFiles;
	string strCurrentPaht = "E:\\data\\uav\\1\\100MSDCF";
	getAllFiles(strCurrentPaht, currentPathFiles);
	for (auto t:currentPathFiles)
	{
		std::cout << t << endl;
	}









}
