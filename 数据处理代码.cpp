#include <iostream>
#include <fstream>
#include <cassert>
#include <string>
#include <vector>
#include <sstream>
#include <cstdio>
#include <io.h>
using namespace std;

const int INF = 2147483647;
ofstream fout;

class Data {
	vector <string> save_name_head;
	vector <string> filename;
	string drive;
	string all_save_name;
	vector <int> data;
	int data_num;
	int filenum;

public:
	//构造函数：参数为文件路径
	Data(vector <string> name, vector <string> savename,string drivetmp,int filenum) {
		drive = drivetmp;
		//filename = drive + name;
		//save_name_head = drive + savename;
		all_save_name = drive + "所有测试数据.bin";
	}

	//读取函数：读取text文件，保存数据数量至data_num
	void readtext_int() {
		ifstream fin(filename);
		int s;

		if (!data.empty())
			data.clear();

		while (fin >> s)
			data.push_back(s);
		
		data_num = data.size();
		fin.close();
	}
	void readtext_float() {
		ifstream fin(filename);
		string s;

		if (!data.empty())
			data.clear();

		while (fin >> s)
			data.push_back((int)(atof(s.c_str()) * 100));

		data_num = data.size();
		fin.close();
	}

	//输出函数：把读入的数据输出
	void display() {
		cout << endl;
		cout << "dispaly:\n";
		for (int i = 0; i < data_num; i++) {
			cout << data[i] << "  ";
			if ((i + 1) % 10 == 0)	cout << endl;
		}
		cout << endl << endl << data_num << endl;
	}

	//数据处理函数：把数据按照 batchsize天+涨幅（lable） 的进行划分并保存
	void PerplusDay_train(float percent,int batchsize) {
		char* tmpc;
		int count = data_num - batchsize;
		int size = batchsize;

		stringstream ss;
		ss << batchsize;
		string trainname = save_name_head + "_涨幅+" + ss.str() + "天_train_";
		ss.str("");
		ss << (int)(count * percent);
		trainname += ss.str() + "组数据.bin";
		ofstream fout(trainname, ofstream::binary);

		for (int i = 0; i < count * percent; i++) {
			int label = FindPercent7(data[batchsize - 1], data[batchsize]);
			tmpc = (char*)&label;
			fout.write(tmpc, 4);
			batchsize++;
			
			for (int j = 0; j < size; j++) {
				tmpc = (char*)&data[i + j];
				fout.write(tmpc, 4);
			}
		}
		fout.close();
	}
	void PerplusDay_eval(float per,int batchsize) {
		char* tmpc;
		int count = data_num - batchsize;
		int size = batchsize;
		float percent = 1 - per;

		stringstream ss;
		ss << batchsize;
		string evalname =drive + "五个银行数据_涨幅+" + ss.str() + "天_eval.bin";
		ofstream fout;
		FILE *file = fopen(evalname.c_str(),"r");
		if (file) {
			fout.open(evalname, ios::binary | ios::app);
			fclose(file);
		}
		else	fout.open(evalname, ios::binary);


		for (int i = count * percent; i < count; i++) {
			int label = FindPercent7(data[batchsize - 1], data[batchsize]);
			tmpc = (char*)&label;
			fout.write(tmpc, 4);
			batchsize++;

			for (int j = 0; j < size; j++) {
				tmpc = (char*)&data[i + j];
				fout.write(tmpc, 4);
			}
		}
		fout.close();
	}
	void CheckDataRand(float percent, int batchsize,int number) {
		char* tmpc;
		int count = data_num - batchsize;
		int size = batchsize;

		stringstream ss;
		ss << batchsize;
		string trainname = save_name_head + "_涨幅+" + ss.str() + "天_randdata_";
		ss.str("");
		ss << number;
		trainname += ss.str() + "组数据.bin";
		ofstream fout(trainname, ofstream::binary);

		for (int i = 0; i < number; i++) {

			int label = FindPercent7(data[batchsize - 1], data[batchsize]);
			tmpc = (char*)&label;
			cout << label << endl;
			fout.write(tmpc, 4);
			batchsize++;

			for (int j = 0; j < size; j++) {
				cout << data[i + j] << "    ";
				tmpc = (char*)&data[i + j];
				fout.write(tmpc, 4);
			}
			cout << endl;
			//write = false;
		}
		fout.close();
	}
	//12个label
	int FindPercent12(int first, int second) {
		float per = (second*1.0 - first*1.0) / (first*1.0);
		per *= 100;
		int label = 0;

		if (between(8.0, 10.0, per)) return label;
		label++;
		if (between(6.0, 8.0, per))	return label;
		label++;
		if (between(4.0, 6.0, per))	return label;
		label++;
		if (between(2.0, 4.0, per))	return label;
		label++;
		if (between(0.0, 2.0, per))	return label;
		label++;
		if (between(-2.0, 0.0, per)) return label;
		label++;
		if (between(-4.0, -2.0, per)) return label;
		label++;
		if (between(-6.0, -4.0, per)) return label;
		label++;
		if (between(-8.0, -6.0, per)) return label;
		label++;
		if (between(-10.0, -8.0, per)) return label;
		label++;
		cout << per << endl;
		cout << label << endl;
		if (between(INF, 10.0, per)) return label;
		return ++label;
	}
	//11个label
	int FindPercent11(int first,int second) {
		float per = (second*1.0 - first*1.0) / (first*1.0);
		per *= 100;
		int label = 0;

		if (between(8.0, 10.0, per)) return label;
		label++;
		if (between(6.0, 8.0, per))	return label;
		label++;
		if (between(4.0, 6.0, per))	return label;
		label++;
		if (between(2.0, 4.0, per))	return label;
		label++;
		if (between(0.0, 2.0, per))	return label;
		label++;
		if (between(-2.0, 0.0, per)) return label;
		label++;
		if (between(-4.0, -2.0, per)) return label;
		label++;
		if (between(-6.0, -4.0, per)) return label;
		label++;
		if (between(-8.0, -6.0, per)) return label;
		label++;
		if (between(-10.0, -8.0, per)) return label;
		cout << label << endl;
		return ++label;
	}
	//7个label
	int FindPercent7(int first, int second) {
		float per = (second*1.0 - first*1.0) / (first*1.0);
		per *= 100;
		int label = 0;

		//cout << per << "    " << first << "    " << second << endl;
		if (between(4.0, 10.0, per)) return label;
		label++;
		if (between(2.0, 4.0, per))	return label;
		label++;
		if (between(0.0, 2.0, per))	return label;
		label++;
		if (between(-2.0, 0.0, per)) return label;
		label++;
		if (between(-4.0, -2.0, per)) return label;
		label++;
		if (between(-10.0, -4.0, per)) return label;
		return ++label;
	}
	//判断是否在区间内
	bool between(float min, float max,  float data) {
		if (data <= max && data > min)
			return true;
		return false;
	}
};

/*
path 文件目录
files 保存文件名称容器
format 查找保存文件类型
*/
void GetAllFormatFiles(string path, vector<string>& files, string format) {
	//文件句柄      
	long   hFile = 0;
	//文件信息      
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)  //第一次查找    
	{
		do {
			p.assign(fileinfo.name);
			if (p.size()>4)
				if (p.substr(p.size() - 4, p.size()) == format)
					files.push_back(p.assign(fileinfo.name));
		} while (_findnext(hFile, &fileinfo) == 0);

		_findclose(hFile); //结束查找    
	}
}

int main() {
	string datadrive = "F:\\data_pro\\BankData\\src";
	string savedrive = "F:\\data_pro\\BankData\\product";
	vector<string> name;
	vector<string> savename;

	int number = sizeof(name) / sizeof(name[0]);

	for (int i = 0; i < number; i++) {
		FILE *file = fopen((drive + name[i]).c_str(), "r");
		if (!file) {
			cout << "文件不存在！" << endl;
			//fclose(file);
			system("pause");
			return 0;
		}
		fclose(file);
	}

	//Data data(name, savename, drive, number);
	//data.readtext_float();
	//data.PerplusDay_train(0.9, 225);
	//data.PerplusDay_eval(0.1, 225);
	//	data.CheckDataRand(0.9, 225, 50);
	//	data.display();

	system("pause");
	return 0;
}
