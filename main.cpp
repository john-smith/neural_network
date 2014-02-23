#include "includes.h"
#include <iostream>
#include <string>
using namespace std;

void readFile(BackPropagation &network,int *size, int layerSize, const char* fileName);
void writeFile(BackPropagation &network, int *size, int layerSize, const char *fileName);
void learning(double *input, double *output, BackPropagation &network, int count);
void calc(BackPropagation &network, double *input, double *output);

//動作確認用のmain関数
int main() {
	BackPropagation network;

	int size[] = {
		3, 10, 3
	};

	int layerSize = 3;
	
	double input [7][3] = {
		{0, 0, 0},
		{0, 0, 1},
		{0, 1, 0},
		{0, 1, 1},
		{1, 0, 0},
		{1, 0, 1},
		{1, 1, 0}
	};

	double output [7][3] = {
		{1, 1, 1},
		{1, 1, 0},
		{1, 0, 1},
		{1, 0, 0},
		{0, 1, 1},
		{0, 1, 0},
		{0, 0, 1}
	};

	double data[2][3] = {
		{1, 1, 1},
	};

	double alpha = 1;

	int action = 0;

	string str;
	
	network.Initialize(size, Lib::sigmoid, Lib::derivationSigmoid, &alpha, 1, 0.01, layerSize);

	

	while(action != 5) {
		cout << "1 : サンプルの計算" << endl;
		cout << "2 : データの学習" << endl;
		cout << "3 : 学習データの読み込み" << endl;
		cout << "4 : 学習データの書き込み" << endl;
		cout << "5 : 終了" << endl;
		cout << "何をしますか？ > ";

		cin >> action;

		switch(action) {
			case 1:
				cout << "入力データ" << endl;
				for(int i = 0;i < 3;i++) {
					cout << data[0][i] << " ";
				}
				cout << "\n\n";

				calc(network, data[0], data[1]);
	
				cout << "出力結果" << endl;
				for(int i = 0;i < 3;i++) {
					cout << data[1][i] << " ";
				}
				cout << "\n\n";
				break;
			case 2:
				cout << "データを学習しています" << endl;
				for(int i = 0;i < 7;i++) {
					learning(input[i], output[i], network, 10000);
				}
				cout << "データを学習しました";
				cout << "\n\n";
				break;
			case 3:
				cout << "ファイル名 > ";
				cin >> str;
				readFile(network, size, layerSize, str.c_str());
				cout << "\n";
				break;
			case 4:
				cout << "ファイル名 > ";
				cin >> str;
				writeFile(network, size, layerSize, str.c_str());
				cout << "\n";
				break;
			case 5:
				cout << "終了します";
				cout << "\n\n";
				break;
			default:
				cout << "入力が不正です" << endl;
		}
	}

	return 0;
}

void readFile(BackPropagation &network,int *size,int layerSize, const char *fileName) {
	int *readSize = new int [layerSize];
	double ***weight = new double** [layerSize - 1];

	for(int i = 0;i < layerSize;i++) {
		readSize[i] = size[i] + 1;
	}
	
	for(int i = 0;i < layerSize - 1;i++) {
		weight[i] = new double* [readSize[i]];
		for(int j = 0;j < readSize[i];j++) {
			weight[i][j] = new double [readSize[i + 1]];
		}
	}
	
	if(IOFromFile::OpenDataFromFile(fileName, weight)) {
		network.SetWeight(weight);
		cout << "学習データを読み込みました" << endl;
	} else {
		cout << "学習データの読み込みに失敗しました" << endl;
	}

	 for(int i = 0;i < layerSize - 1;i++) {
		for(int j = 0;j < readSize[i];j++) {
			delete [] weight[i][j];
		}
		delete [] weight[i];
	}
	delete [] weight;
	delete [] readSize;
}

void writeFile(BackPropagation &network, int *size,int layerSize, const char *fileName) {
	int *writeSize = new int [layerSize];
	double ***weight = new double** [layerSize - 1];

	for(int i = 0;i < layerSize;i++) {
		writeSize[i] = size[i] + 1;
	}
	
	for(int i = 0;i < layerSize - 1;i++) {
		weight[i] = new double* [writeSize[i]];
		for(int j = 0;j < writeSize[i];j++) {
			weight[i][j] = new double [writeSize[i + 1]];
		}
	}

	network.GetWeight(weight);

	if(IOFromFile::SaveData(fileName, (double***)weight, writeSize, layerSize)) {
		cout << "学習データを書き込みました" << endl;
	} else {
		cout << "学習データの書き込みに失敗しました" << endl;
	}

	for(int i = 0;i < layerSize - 1;i++) {
		for(int j = 0;j < writeSize[i];j++) {
			delete [] weight[i][j];
		}
		delete [] weight[i];
	}
	delete [] weight;
	delete [] writeSize;
}

void learning(double *input, double *output, BackPropagation &network, int count) {
	network.SetInputData(input);
	network.SetOutputData(output);
	for(int i = 0;i < count;i++) {
		network.FeedForward();
		network.CalcError();
		network.BackPropagate();
	}
}

void calc(BackPropagation &network, double *input, double *output) {
	network.SetInputData(input);
	network.FeedForward();
	network.GetResult(output);
}

