#include "includes.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
using namespace std;

bool IOFromFile::OpenDataFromFile(const char* fileName, double ***data) {
	ifstream ifs(fileName);
	string buf;

	int j, k = 0, i = 0;
	size_t pos;

	if(ifs.fail()) {
		return false;
	}

	while(ifs && getline(ifs, buf)) {
		if(buf.length() == 0) {
			k++;
			i = 0;
		} else {
			j = 0;
			while((pos = buf.find(",")) != string::npos) {
				data[k][i][j++] = atof(buf.c_str());
				buf = buf.substr(pos + 1);
			}
			
			data[k][i++][j] = atof(buf.c_str());
		}
	}

	return true;
}

bool IOFromFile::SaveData(const char* fileName, double ***data, int *size, int layerSize, int dataNum) {
	ofstream ofs(fileName, ios::trunc);
	
	if(ofs) {
		for(int l = 0;l < dataNum;l++) {
			for(int i = 0;i < layerSize - 1;i++) {
				for(int j = 0;j < size[i];j++) {
					for(int k = 0;k < size[i + 1];k++) {
						ofs << data[i][j][k];
						if(k < size[i + 1] - 1) {
							ofs << ",";
						}
					}
					ofs << endl;
				}
				ofs << endl;
			}
		}
	} else {
		return false;
	}

	return true;
}

	