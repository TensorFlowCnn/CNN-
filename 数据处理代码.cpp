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

public:
	//构造函数：参数为文件路径
	Data(string name,string savename,string drivetmp) {
		drive = drivetmp;
		filename = drive + name;
		save_name_head = drive + savename;
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
		cout << "dispaly:\n";
		for (int i = 0; i < data_num; i++) {
			cout << data[i] << "  ";
			if ((i + 1) % 10 == 0)	cout << endl;
		}
		cout << endl << endl << data_num << endl;
	}

	//数据处理函数：把数据按照 五周+一天（lable） 的大端格式进行划分并保存
	void ODplusFW() {
		char* tmpc;
		int index = 25, i;
		int count = data_num - 25;
		stringstream ss;
		ss << (int)(count*0.9);
		string trainname = save_name_head + "_一天+五周_train_" + ss.str() + "组数据.bin";
		ss.str("");
		ss << (count - (int)(count*0.9));
		string evalname = save_name_head + "_一天+五周_eval_" + ss.str() + "组数据.bin";
		ofstream fout1(trainname, ofstream::binary);
		ofstream fout2(evalname, ofstream::binary);

		for (i = 0; i < count*0.9; i++) {
			tmpc = (char*)&data[index];
			fout1.write(tmpc, 4);
			index++;

			for (int j = 0; j < 25; j++) {
				tmpc = (char*)&data[i + j];
				fout1.write(tmpc, 4);
			}
		}
		fout1.close();

		for (; i < count; i++) {
			tmpc = (char*)&data[index];
			fout2.write(tmpc, 4);
			index++;

			for (int j = 0; j < 25; j++) {
				tmpc = (char*)&data[i + j];
				fout2.write(tmpc, 4);
			}
		}
		fout2.close();
	}

	//数据处理函数：把数据按照 五周+升降幅度百分比（lable） 的大端格式进行划分并保存
	void PerplusFW() {
		char* tmpc;
		int index = 25, i;
		int count = data_num - 25;
		stringstream ss;
		ss << (int)(count*0.9);
		string trainname = save_name_head + "_涨幅+五周_train_" + ss.str() + "组数据.bin";
		ss.str("");
		ss << (count - (int)(count*0.9));
		string evalname = save_name_head + "_涨幅+五周_eval_" + ss.str() + "组数据.bin";
		ofstream fout1(trainname, ofstream::binary);
		ofstream fout2(evalname, ofstream::binary);

		for (i = 0; i < count*0.9; i++) {
			int label = FindPercent(data[index - 1], data[index]);
			tmpc = (char*)&label;
			fout1.write(tmpc, 4);
			index++;

			for (int j = 0; j < 25; j++) {
				tmpc = (char*)&data[i + j];
				fout1.write(tmpc, 4);
			}
		}
		fout1.close();

		for (; i < count; i++) {
			int label = FindPercent(data[index - 1], data[index]);
			tmpc = (char*)&label;
			fout2.write(tmpc, 4);
			index++;

			for (int j = 0; j < 25; j++) {
				tmpc = (char*)&data[i + j];
				fout2.write(tmpc, 4);
			}
		}
		fout2.close();
	}
	int FindPercent(int first,int second) {
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
		label++;
		if (between(INF, 10.0, per)) return label;
		label++;
		if (between(-10.0, -INF, per)) return label;
	}
	//判断是否在区间内
	bool between(float min, float max,  float data) {
		if (data <= max && data > min)
			return true;
		return false;
	}
};

int main() {
	string drive = "F:\\data_pro\\";
	string name = "data_for_test_since1991.txt";
	string savename = "平安银行_since1991";

	FILE *file=fopen((drive+name).c_str(),"r");
	if (!file) {
		cout << "文件不存在！" << endl;
		fclose(file);
		return 0;
	}
	fclose(file);

	Data data(name,savename,drive);
	data.readtext_int();
//	data.display();
	data.ODplusFW();
	data.PerplusFW();

	system("pause");
	return 0;
}
