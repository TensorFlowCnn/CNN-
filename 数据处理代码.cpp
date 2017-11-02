#include <iostream>
#include <fstream>
#include <cassert>
#include <string>
#include <vector>
#include <sstream>
#include <cstdio>
using namespace std;

const int INF = 2147483647;

class Data {
	string save_name_head;
	string filename;
	string drive;
	vector <int> data;
	int data_num;
	int labelnum[12];

public:
	//构造函数：参数为文件路径
	Data(string name,string savename,string drivetmp) {
		drive = drivetmp;
		filename = drive + name;
		save_name_head = drive + savename;
		for (int i = 0; i < 12; i++)
			labelnum[i] = 0;
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
		/*for (int i = 0; i < 12; i++)
			cout << labelnum[i] << endl;*/
	}

	//数据处理函数：把数据按照 五周+升降幅度百分比（lable） 的大端格式进行划分并保存
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
		string evalname = save_name_head + "_涨幅+" + ss.str() + "天_train_";
		ss.str("");
		ss << (count - (int)(count * percent));
		evalname += ss.str() + "组数据.bin";
		ofstream fout(evalname, ofstream::binary);

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
	//11个label
	int FindPercent11(int first,int second) {
		float per = (second*1.0 - first*1.0) / (first*1.0);
		per *= 100;
		int label = 0;

		//cout << per << "    " << first << "    " << second << endl;
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
		//cout << label << endl;
		return ++label;
	}
	//判断是否在区间内
	bool between(float min, float max,  float data) {
		if (data <= max && data > min)
			return true;
		return false;
	}
};

int main() {
	string drive = "F:\\data_pro\\BankData\\";
	string name = "data_since1991_pinanyinhang_float.txt";
	string savename = "农业银行_since2010";

	FILE *file=fopen((drive+name).c_str(),"r");
	if (!file) {
		cout << "文件不存在！" << endl;
		fclose(file);
		return 0;
	}
	fclose(file);

	Data data(name,savename,drive);
	data.readtext_float();
	data.PerplusDay_train(0.9,225);
	data.PerplusDay_eval(0.1,225);
//	data.display();

	system("pause");
	return 0;
}
