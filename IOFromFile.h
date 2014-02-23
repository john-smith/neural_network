#ifndef __IO_FROM_FILE_H__
#define __IO_FROM_FILE_H__

#include <string>

class IOFromFile {
public:
	static bool OpenDataFromFile(const char* fileName, double ***data);
	static bool SaveData(const char* filename, double*** weight, int *size, int layerSize, int dataNum = 1);
};

#endif