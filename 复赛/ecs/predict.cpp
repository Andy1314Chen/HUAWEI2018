#include "predict.h"
// #include "ga.h"
#include "genneticAlgorithm.h"
#include "mtm.h"
#include <stdio.h>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <ctime>
#include <locale>
#include <iomanip>
#include <map>
#include <numeric>

using namespace std;

// 測試集信息結構體
struct TestInfo
{
	// server number
	int serverNum;
	// CPU number of the server  G: General; L: Large-Memory; H: High-Performance
	std::vector<int> CPU;
	std::vector<int> MEM;
	std::vector<string> serverName;
	int G_CPU;
	int L_CPU;
	int H_CPU;
	// MEM number of the server
	int G_MEM;
	int L_MEM;
	int H_MEM;
	// flavor number
	int predictNum;
	// is CPU?
	int isCPU;
	// the first day
	long firstDay;
	// the end day
	long endDay;
	// delta date
	int deltaDate;
	// dayOfWeek
	vector<int> dayOfWeek;
};

// 訓練集信息結構體
struct TrainInfo
{
	// flavor of train
	vector<int> flavor;
	// date of train
	vector<long> date;
	vector<string> date2;
	// day of train 去重
	// day of week
	vector<int> dayOfWeek;
	vector<long> day;
	vector<string> day2;
};

// 待預測虛擬機規格信息
struct PredInfo
{
	// flavor to predict
	vector<int> predFlavor;
	// cpu number of flavor
	vector<int> predCPU;
	// mem number of flavor
	vector<int> predMEM;
};

// flavorX的訓練數據
struct TrainFlavor
{
	// flavorX
	int flavorX;
	// history data
	vector<int> histData;
	// datetime
	vector<long> histDatetime;
	// dayOfWeek
	vector<int> dayOfWeek;
};

// 因子
struct WeekFactor
{
	// day of week
	int dayOfWeek;
	// factor list
	std::vector<float> factor;
	// median factor
	float medianFactor;
	// mean factor
	float meanFactor;
	// mix factor
	float mixFactor;
};

// 字符串分割算法
vector<string> &split(const string &str, char delim, vector<string> &elems, bool skip_empty = true)
{
	elems.clear();

	istringstream iss(str);
	for (string item; getline(iss, item, delim);)
		if (skip_empty && item.empty())
			continue;
		else
			elems.push_back(item);

	return elems;
}

// return the day_of_week of the date
int computeWeek(string date)
{
	vector<string> res;
	int year, month, day;
	split(date, '-', res);
	year = stoi(res[0]); //
	month = stoi(res[1]);
	day = stoi(res[2]);

	static int t[] = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
	year -= month < 3;
	int week = (year + year / 4 - year / 100 + year / 400 + t[month - 1] + day) % 7;

	return week > 0 ? week : 7;
}

// convert the datetime to the day numbers
long g(string date)
{
	vector<string> res;
	int y, m, d;
	split(date, '-', res);
	y = stoi(res[0]); //
	m = stoi(res[1]);
	d = stoi(res[2]);

	m = (m + 9) % 12;
	y = y - m / 10;
	return 365 * y + y / 4 - y / 100 + y / 400 + (m * 306 + 5) / 10 + (d - 1);
}

// convert the day numbers to datetime string
string d(long g)
{
	long y, ddd, mi;

	y = (10000 * g + 14780) / 3652425;
	ddd = g - (y * 365 + y / 4 - y / 100 + y / 400);
	if (ddd < 0)
	{
		y--;
		ddd = g - (y * 365 + y / 4 - y / 100 + y / 400);
	}
	mi = (52 + 100 * ddd) / 3060;
	int year, month, day;
	year = y + (mi + 2) / 12;
	month = (mi + 2) % 12 + 1;
	day = ddd - (mi * 306 + 5) / 10 + 1;

	return to_string(year) + "-" + to_string(month) + "-" + to_string(day);
}

// product a vector<int> which include day of week from date1 to date2
std::vector<int> productDateVector(string date1, int days)
{
	string strTmp = date1;
	std::vector<int> intRes;

	intRes.push_back(computeWeek(strTmp));
	while (days--)
	{
		strTmp = d(g(strTmp) + 1);
		intRes.push_back(computeWeek(strTmp));
	}
	return intRes;
}

// first fit
vector<ResultNode> firstFit(vector<int> flavor, vector<int> cpuNum, vector<int> memNum, int CPU, int MEM)
{
	int res = 0, n = flavor.size();
	int remCPU[n], remMEM[n];

	vector<string> resultString;
	vector<ResultNode> result;

	for (int i = 0; i < n; i++)
	{
		int j;
		for (j = 0; j < res; j++) // 服務器還有位置
		{
			if ((remCPU[j] >= cpuNum[i]) && (remMEM[j] >= memNum[i]))
			{
				remCPU[j] -= cpuNum[i];
				remMEM[j] -= memNum[i];

				resultString[j] += " flavor" + to_string(flavor[i]);

				if ((result[j].remCPU >= cpuNum[i]) && (result[j].remMEM >= memNum[i]))
				{
					result[j].remCPU -= cpuNum[i];
					result[j].remMEM -= memNum[i];
					result[j].flavor.push_back(flavor[i]);
					result[j].cpuNumber.push_back(cpuNum[i]);
					result[j].memNumber.push_back(memNum[i]);
				}

				break;
			}
		}
		if (j == res) // 開闢新服務器
		{
			remCPU[res] = CPU - cpuNum[i];
			remMEM[res] = MEM - memNum[i];
			resultString.push_back("\nflavor" + to_string(flavor[i]));
			res++;

			ResultNode tmpNode;

			tmpNode.remCPU = CPU - cpuNum[i];
			tmpNode.remMEM = MEM - memNum[i];
			tmpNode.flavor.push_back(flavor[i]);	// flavor
			tmpNode.cpuNumber.push_back(cpuNum[i]); // cpu number
			tmpNode.memNumber.push_back(memNum[i]); // mem number

			result.push_back(tmpNode);
		}
	}

	return result;
}

// First Fit Decreasing algorithm
vector<ResultNode> firstFitDecreseing(vector<int> flavor, vector<int> cpuNum, vector<int> memNum, int CPU, int MEM, int flag)
{
	vector<pair<int, int>> vectTmp1;
	vector<pair<int, int>> vectTmp2;
	int n = flavor.size();
	// cpu
	if (!flag)
	{
		for (int i = 0; i < n; i++)
		{
			vectTmp1.push_back(make_pair(cpuNum[i], flavor[i]));
			vectTmp2.push_back(make_pair(cpuNum[i], memNum[i]));
		}
		// sort
		sort(vectTmp1.rbegin(), vectTmp1.rend());
		sort(vectTmp2.rbegin(), vectTmp2.rend());
		for (int i = 0; i < n; i++)
		{
			cpuNum[i] = vectTmp1[i].first;
			flavor[i] = vectTmp1[i].second;
			memNum[i] = vectTmp2[i].second;
		}
		return firstFit(flavor, cpuNum, memNum, CPU, MEM);
	}
	// mem
	else
	{
		for (int i = 0; i < n; i++)
		{
			vectTmp1.push_back(make_pair(memNum[i], flavor[i]));
			vectTmp2.push_back(make_pair(memNum[i], cpuNum[i]));
		}
		// sort
		sort(vectTmp1.rbegin(), vectTmp1.rend());
		sort(vectTmp2.rbegin(), vectTmp2.rend());
		for (int i = 0; i < n; i++)
		{
			memNum[i] = vectTmp1[i].first;
			flavor[i] = vectTmp1[i].second;
			cpuNum[i] = vectTmp2[i].second;
		}
		return firstFit(flavor, cpuNum, memNum, CPU, MEM);
	}
}

// 预测结果后处理 骚操作
std::vector<ResultNode> spSolution(std::vector<int> maxCPU, std::vector<int> maxMEM, std::vector<FlavorX> gaFlavorX, vector<int> &reqFlavor)
{

	std::vector<ResultNode> resFirst = knapsackFit(maxCPU, maxMEM, gaFlavorX);

	for (unsigned int i = 0; i < resFirst.size(); i++)
	{
		if (resFirst[i].radio > 55.0) // 利用率大于60，则填满
		{
			while ((resFirst[i].remMEM >= gaFlavorX[0].mem) && (resFirst[i].remCPU >= gaFlavorX[0].cpu)) // 默认虚拟机在最前面的规格最小
			{
				for (unsigned int k = 0; k < gaFlavorX.size(); k++)
				{
					if ((resFirst[i].remMEM >= gaFlavorX[k].mem) && (resFirst[i].remCPU >= gaFlavorX[k].cpu))
					{
						resFirst[i].remMEM -= gaFlavorX[k].mem;
						resFirst[i].remCPU -= gaFlavorX[k].cpu;

						resFirst[i].flavor.push_back(gaFlavorX[k].flavor);
						resFirst[i].cpuNumber.push_back(gaFlavorX[k].cpu);
						resFirst[i].memNumber.push_back(gaFlavorX[k].mem);

						gaFlavorX[k].totalNum += 1;
						reqFlavor[k] += 1;
					}
				}
			}
		}
		else // 利用率低，把服务器去掉
		{
			for (unsigned int k = 0; k < resFirst[i].flavor.size(); k++)
			{
				for (unsigned int j = 0; j < gaFlavorX.size(); j++)
				{
					if (gaFlavorX[j].flavor == resFirst[i].flavor[k]) // 把服务器里放置的虚拟机都去掉
						reqFlavor[j] -= 1;
				}
			}
			resFirst.erase(resFirst.begin() + i); // 去掉该台服务器
		}
	}

	return resFirst;
}

// save the result
void saveResult(TestInfo test, PredInfo pred, vector<int> reqFlavor, char *filename)
{
	vector<int> needFlavor; // name of the flavor need to pack
	vector<int> needCPU;	// cpu of the flavor need to pack
	vector<int> needMEM;	// mem of the flavor need to pack

	vector<FlavorX> gaFlavor;
	int gaTotalNum = 0;

	for (int i = 0; i < test.predictNum; i++)
	{
		FlavorX tmp;
		tmp.flavor = pred.predFlavor[i];
		tmp.cpu = pred.predCPU[i];
		tmp.mem = pred.predMEM[i];
		tmp.totalNum = reqFlavor[i];
		gaTotalNum += reqFlavor[i];
		gaFlavor.push_back(tmp);
	}

	// 对虚拟机分类讨论，MEM/CPU = {1,2,4}

	// 遗传算法
	// vector<ResultNode> result = gaFit(50, 0.8, 0.5, test.CPU, test.MEM, gaFlavor, gaTotalNum);
	// 动态规划
	// vector<ResultNode> result = knapsackFit(test.CPU, test.MEM, gaFlavor);
	// 骚操作
	vector<ResultNode> result = spSolution(test.CPU, test.MEM, gaFlavor, reqFlavor);
	int serverNum = result.size();

	// double f1_CPU = 0, f1_MEM = 0, f2_CPU = 0, f2_MEM = 0;
	std::vector<int> tmpServer = {0, 0, 0};
	for (int i = 0; i < serverNum; i++)
	{
		// f2_CPU += test.CPU[result[i].serverClass];
		// f2_MEM += test.MEM[result[i].serverClass];
		// f1_CPU += test.CPU[result[i].serverClass] - result[i].remCPU;
		// f1_MEM += test.MEM[result[i].serverClass] - result[i].remMEM;
		switch (result[i].serverClass)
		{
		case 0:
			tmpServer[0]++;
			break;
		case 1:
			tmpServer[1]++;
			break;
		case 2:
			tmpServer[2]++;
			break;
		}
	}
	// printf("radio %.2f\n", f1_CPU * 50.0 / f2_CPU + f1_MEM * 50.0 / f2_MEM);
	// predict total flavor number
	int totalFlavor = accumulate(reqFlavor.begin(), reqFlavor.end(), 0);
	// result string
	string outputStr = to_string(totalFlavor) + "\n";
	for (int i = 0; i < test.predictNum; i++)
	{
		outputStr += "flavor" + to_string(pred.predFlavor[i]) + " " + to_string(reqFlavor[i]) + "\n";
	}

	for (int k = 0; k < test.serverNum; k++)
	{
		if (tmpServer[k])
		{ // 写入第一台服务器
			outputStr += "\n" + test.serverName[k] + " " + to_string(tmpServer[k]);
			for (int i = 0, cnt = 1; i < serverNum; i++)
			{
				if (result[i].serverClass == k)
				{
					std::vector<int> flavorTmp;
					flavorTmp.assign(result[i].flavor.begin(), result[i].flavor.end());
					sort(flavorTmp.begin(), flavorTmp.end());
					flavorTmp.erase(unique(flavorTmp.begin(), flavorTmp.end()), flavorTmp.end());

					outputStr += "\n" + test.serverName[k] + "-" + to_string(cnt++) + " ";
					for (unsigned int j = 0; j < flavorTmp.size(); j++)
					{
						int n = count(result[i].flavor.begin(), result[i].flavor.end(), flavorTmp[j]);
						outputStr += "flavor" + to_string(flavorTmp[j]) + " " + to_string(n) + " ";
					}
				}
			}
			outputStr += "\n";
		}
	}

	write_result(outputStr.c_str(), filename);
}

float median(std::vector<float> v)
{
	unsigned int n = v.size() / 2;
	std::nth_element(v.begin(), v.begin() + n, v.end());
	float vn = v[n];
	if (v.size() % 2 == 1)
	{
		return vn;
	}
	else
	{
		std::nth_element(v.begin(), v.begin() + n - 1, v.end());
		return 0.5 * (vn + v[n - 1]);
	}
}

// time regular 4 一次指数平滑滤波
void timeRegular4(TestInfo test, TrainInfo train, PredInfo pred, char *filename)
{
	vector<int> reqFlavor;
	vector<int> reqFlavor2;
	vector<int> reqFlavor3;
	vector<TrainFlavor> trainFlavorX;

	for (int i = 0; i < test.predictNum; i++)
	{
		TrainFlavor tmp;
		tmp.flavorX = pred.predFlavor[i];
		for (unsigned int j = 0; j < train.flavor.size(); j++)
		{
			if (train.flavor[j] == pred.predFlavor[i])
			{
				tmp.histDatetime.push_back(train.date[j]);
			}
		}
		trainFlavorX.push_back(tmp);
	}

	int n = train.day.size();
	int during = train.day[n - 1] - train.day[0] + 1; // days of the train data set

	for (int k = 0; k < test.predictNum; k++)
	{
		for (int i = 0, j = 0; (i < during) && (j < n); i++)
		{
			int cnt = 0;
			if ((train.day[0] + i) == train.day[j])
			{
				cnt = count(trainFlavorX[k].histDatetime.begin(), trainFlavorX[k].histDatetime.end(), train.day[j++]);
				trainFlavorX[k].histData.push_back(cnt);
			}
			else
				trainFlavorX[k].histData.push_back(0); // 确实值填充为0
		}
	}

	// // 濾波處理，主要是將極大值點降低一些
	// for (int k = 0; k < test.predictNum; k++)
	// {
	// 	double sum = accumulate(trainFlavorX[k].histData.begin(), trainFlavorX[k].histData.end(), 0.0);
	// 	double mean = sum / trainFlavorX[k].histData.size();
	// 	double sq_sum = inner_product(trainFlavorX[k].histData.begin(), trainFlavorX[k].histData.end(), trainFlavorX[k].histData.begin(), 0.0);
	// 	double stdev = sqrt(sq_sum / trainFlavorX[k].histData.size() - mean * mean);

	// 	for (unsigned int i = 0; i < trainFlavorX[k].histData.size(); i++)
	// 	{
	// 		if (abs(trainFlavorX[k].histData[i] - mean) / (stdev) > 4.0)
	// 		{
	// 			 trainFlavorX[k].histData[i] = (int)(trainFlavorX[k].histData[i] * 0.95); //(int)(mean + 5); // 把極大值降低一些
	// 		}
	// 	}
	// }

	for (int i = 0; i < test.predictNum; i++)
	{
		auto endIter = trainFlavorX[i].histData.end();
		auto beginIter = trainFlavorX[i].histData.end() - test.deltaDate;

		std::vector<float> S;
		std::vector<float> originalY;
		while (beginIter >= trainFlavorX[i].histData.begin())
		{
			double tmp = std::accumulate(beginIter, endIter, 0);
			originalY.push_back(tmp);

			endIter = beginIter - 1;
			beginIter = endIter - test.deltaDate;
		}
		std::reverse(originalY.begin(), originalY.end());

		float alpha = 0.22, S0 = 0;
		S0 = (originalY[0] + originalY[1]) / 2.0;
		S.push_back(S0);
		for (unsigned int k = 0; k < originalY.size(); k++)
		{
			double tmp = S[k] * (1 - alpha) + alpha * originalY[k]; // 正确的公式是 S(t+1) = (1-alpha)*S(t) + alpha * Y(t)
			S.push_back(tmp);
		}

		float val = originalY.back() * alpha + (1 - alpha) * S.back(); //; //alpha *
		reqFlavor.push_back((int)(val));
	}

	// for (unsigned int i = 0; i < reqFlavor.size(); i++)
	// {
	// 	if (reqFlavor[i] > 400)
	// 		reqFlavor[i] = (int)(reqFlavor[i] * 2.8);
	// 	else if (reqFlavor[i] > 100)
	// 		reqFlavor[i] = (int)(reqFlavor[i]);
	// 	else
	// 		reqFlavor[i] = (int)(reqFlavor[i] * 2);
	// }

	saveResult(test, pred, reqFlavor, filename);
}

// time regular 3 提取总的周期因子
void timeRegular3(TestInfo test, TrainInfo train, PredInfo pred, char *filename)
{
	// flavor of train
	vector<int> flavor;
	// date of train
	vector<long> date;
	vector<string> date2;

	// 對flavor1-flavor18規則求和，提取總周期因子
	for (unsigned int i = 0; i < train.flavor.size(); i++)
	{ // delete the flavorX > 15
		flavor.push_back(train.flavor[i]);
		date.push_back(train.date[i]);
		date2.push_back(train.date2[i]);
	}
	train.flavor = flavor;
	train.date = date;
	train.date2 = date2;

	// during:训练集时间跨度, n:训练集实际天数，during>=n
	int n = train.day.size();
	int during = train.day[n - 1] - train.day[0] + 1;

	TrainFlavor histTrain;
	// 统计训练集每天总的请求量，缺失天数填0处理，并附上曜日
	for (int i = 0, j = 0; (i < during) && (j < n); i++)
	{
		int cnt = 0;
		if ((train.day[0] + i) == train.day[j])
		{
			cnt = count(train.date.begin(), train.date.end(), train.day[j++]);
			histTrain.histData.push_back(cnt);
			histTrain.dayOfWeek.push_back(train.dayOfWeek[i]);
		}
		else
		{
			histTrain.histData.push_back(0);
			histTrain.dayOfWeek.push_back(train.dayOfWeek[i]);
		}
	}

	// 濾波處理，主要是將極大值點降低一些
	double sum0 = std::accumulate(histTrain.histData.begin(), histTrain.histData.end(), 0.0);
	double mean0 = sum0 / histTrain.histData.size();
	double sq_sum0 = inner_product(histTrain.histData.begin(), histTrain.histData.end(), histTrain.histData.begin(), 0.0);
	double stdev0 = sqrt(sq_sum0 / histTrain.histData.size() - mean0 * mean0);
	for (unsigned int i = 0; i < histTrain.histData.size(); i++)
	{
		if (abs(histTrain.histData[i] - mean0) / (stdev0) > 2.5)
		{
			// histTrain.histData[i] = (int)(histTrain.histData[i] * 0.55);
		}
	}

	// 存放周期因子, 总请求量
	std::vector<WeekFactor> medianFactor;
	for (int i = 0; i < 7; i++)
	{
		// 存放第N周的因子
		WeekFactor tmp;
		tmp.dayOfWeek = i + 1;
		medianFactor.push_back(tmp);
	}

	std::vector<float> numList;
	std::vector<int> weekList;
	for (unsigned int i = 0; i < histTrain.histData.size(); i++)
	{

		numList.push_back(histTrain.histData[i]);
		weekList.push_back(histTrain.dayOfWeek[i]);
		if (histTrain.dayOfWeek[i] == 7)
		{
			float sum = std::accumulate(numList.begin(), numList.end(), 0.0);
			float mean = sum / numList.size();

			for (unsigned int j = 0; j < numList.size(); j++)
			{
				// 存放曜日因子
				medianFactor[weekList[j] - 1].factor.push_back(numList[j] / mean);
			}
			numList.clear();
			weekList.clear();
		}
	}
	// 周期因子包括中衛數因子，均值因子和融合因子
	for (int i = 0; i < 7; i++)
	{
		float medianTmp = median(medianFactor[i].factor);
		float meanTmp = std::accumulate(medianFactor[i].factor.begin(), medianFactor[i].factor.end(), 0.0) / medianFactor[i].factor.size();
		// 中位数因子
		medianFactor[i].medianFactor = medianTmp;
		// 均值因子
		medianFactor[i].meanFactor = meanTmp;
		// 融合因子
		medianFactor[i].mixFactor = medianTmp * 0.5 + meanTmp * 0.5;
	}

	vector<int> reqFlavor;
	vector<int> reqFlavor2;
	vector<TrainFlavor> trainFlavorX;
	// 需要預測的flaovr規格歷史數據
	for (int i = 0; i < test.predictNum; i++)
	{
		TrainFlavor tmp;
		tmp.flavorX = pred.predFlavor[i];
		for (unsigned int j = 0; j < train.flavor.size(); j++)
		{
			if (train.flavor[j] == pred.predFlavor[i])
			{
				tmp.histDatetime.push_back(train.date[j]);
			}
		}
		trainFlavorX.push_back(tmp);
	}
	// 統計歷史上每天的請求量，缺失值填充爲0
	for (int k = 0; k < test.predictNum; k++)
	{
		for (int i = 0, j = 0; (i < during) && (j < n); i++)
		{
			int cnt = 0;
			if ((train.day[0] + i) == train.day[j])
			{
				cnt = count(trainFlavorX[k].histDatetime.begin(), trainFlavorX[k].histDatetime.end(), train.day[j++]);
				trainFlavorX[k].histData.push_back(cnt);
				trainFlavorX[k].dayOfWeek.push_back(train.dayOfWeek[i]);
			}
			else
			{
				trainFlavorX[k].histData.push_back(0);
				trainFlavorX[k].dayOfWeek.push_back(train.dayOfWeek[i]);
			}
		}
	}
	// 取最後10天的均值作爲base, 乘以周期因子再進行預測
	for (int i = 0; i < test.predictNum; i++)
	{
		// last n days
		float base = 0;
		for (int j = 0; j < 10; j++)
		{
			int num = trainFlavorX[i].histData[during - 1 - j];
			int weekday = trainFlavorX[i].dayOfWeek[during - 1 - j];
			base = base + num / medianFactor[weekday - 1].mixFactor;
		}
		base = base / 10;

		float res = 0;
		for (unsigned int k = 0; k < test.dayOfWeek.size(); k++)
		{
			res = res + base * medianFactor[test.dayOfWeek[k] - 1].mixFactor;
		}
		reqFlavor.push_back((int)(res)); // (test.deltaDate / 7) *
	}

	// 濾波處理，主要是將極大值點降低一些
	for (int k = 0; k < test.predictNum; k++)
	{
		double sum = accumulate(trainFlavorX[k].histData.begin(), trainFlavorX[k].histData.end(), 0.0);
		double mean = sum / trainFlavorX[k].histData.size();
		double sq_sum = inner_product(trainFlavorX[k].histData.begin(), trainFlavorX[k].histData.end(), trainFlavorX[k].histData.begin(), 0.0);
		double stdev = sqrt(sq_sum / trainFlavorX[k].histData.size() - mean * mean);

		for (unsigned int i = 0; i < trainFlavorX[k].histData.size(); i++)
		{
			if (abs(trainFlavorX[k].histData[i] - mean) / (stdev) > 2.0)
			{
				trainFlavorX[k].histData[i] = (int)(trainFlavorX[k].histData[i] * 0.65); //(int)(trainFlavorX[k].histData[i] * 0.65); // 把極大值降低一些
			}
		}
	}

	for (int i = 0; i < test.predictNum; i++)
	{
		int tmp = 0;
		for (int j = 0; j < test.deltaDate; j++)
		{
			tmp += trainFlavorX[i].histData[during - 1 - j];
		}
		reqFlavor2.push_back(tmp);
	}

	// for (unsigned int i = 0; i < reqFlavor.size(); i++)
	// {
	// 	reqFlavor[i] = (int)((reqFlavor[i] * 0.35 + reqFlavor2[i] * 0.7)); // * 0.35 + reqFlavor2[i] * 0.7 83.215 0.35:0.7
	// }

	saveResult(test, pred, reqFlavor, filename);
}

// time regular 2 单个虚拟机规格周期因子提取
void timeRegular2(TestInfo test, TrainInfo train, PredInfo pred, char *filename)
{
	std::vector<int> reqFlavor;
	std::vector<TrainFlavor> trainFlavorX;
	// 提取待预测虚拟机的历史信息
	for (int i = 0; i < test.predictNum; i++)
	{
		TrainFlavor tmp;
		tmp.flavorX = pred.predFlavor[i];
		for (unsigned int j = 0; j < train.flavor.size(); j++)
		{
			if (train.flavor[j] == pred.predFlavor[i])
			{
				tmp.histDatetime.push_back(train.date[j]);
			}
		}
		trainFlavorX.push_back(tmp);
	}

	int n = train.day.size();
	int during = train.day[n - 1] - train.day[0] + 1; // days of the train data set
	// 缺失值填0,并附加曜日
	for (int k = 0; k < test.predictNum; k++)
	{
		for (int i = 0, j = 0; (i < during) && (j < n); i++)
		{
			int cnt = 0;
			if ((train.day[0] + i) == train.day[j])
			{
				cnt = count(trainFlavorX[k].histDatetime.begin(), trainFlavorX[k].histDatetime.end(), train.day[j++]);
				trainFlavorX[k].histData.push_back(cnt);
				trainFlavorX[k].dayOfWeek.push_back(train.dayOfWeek[i]); // 曜日
			}
			else
			{
				trainFlavorX[k].histData.push_back(0); // 缺失值填0
				trainFlavorX[k].dayOfWeek.push_back(train.dayOfWeek[i]);
			}
		}
	}

	for (int k = 0; k < test.predictNum; k++)
	{
		// 存放周期因子
		std::vector<WeekFactor> medianFactor;
		for (int i = 0; i < 7; i++)
		{
			WeekFactor tmp;
			tmp.dayOfWeek = i + 1;
			medianFactor.push_back(tmp);
		}

		std::vector<float> numList;
		std::vector<int> weekList;
		for (unsigned int i = 0; i < trainFlavorX[k].histData.size(); i++)
		{

			numList.push_back(trainFlavorX[k].histData[i]);
			weekList.push_back(trainFlavorX[k].dayOfWeek[i]);
			if (trainFlavorX[k].dayOfWeek[i] == 7)
			{
				float sum = std::accumulate(numList.begin(), numList.end(), 0.0);
				float mean = sum / numList.size() > 0.5 ? sum / numList.size() : 0.5; // max(sum / numList.size(), 1); // 为了放置mean为0

				for (unsigned int j = 0; j < numList.size(); j++)
				{
					medianFactor[weekList[j] - 1].factor.push_back(numList[j] / mean);
				}
				numList.clear();
				weekList.clear();
			}
		}
		// 周期因子包括中衛數因子，均值因子和融合因子
		for (int i = 0; i < 7; i++)
		{
			float medianTmp = median(medianFactor[i].factor);
			float meanTmp = std::accumulate(medianFactor[i].factor.begin(), medianFactor[i].factor.end(), 0.0) / medianFactor[i].factor.size();
			medianFactor[i].medianFactor = medianTmp;
			medianFactor[i].meanFactor = meanTmp;
			medianFactor[i].mixFactor = medianTmp * 0.5 + meanTmp * 0.5; //, 0.01);max(
		}

		// last n days
		float base = 0;
		for (int j = 0; j < 10; j++)
		{
			int num = trainFlavorX[k].histData[during - 1 - j];
			int weekday = trainFlavorX[k].dayOfWeek[during - 1 - j];
			base = base + num; // / medianFactor[weekday - 1].mixFactor
		}
		base = base / 10;

		float res = 0;
		for (unsigned int i = 0; i < test.dayOfWeek.size(); i++)
		{
			res = res + base * medianFactor[test.dayOfWeek[i] - 1].mixFactor;
		}

		reqFlavor.push_back((int)(res)); // (test.deltaDate / 7) * 35 //+ 35
	}
	for (unsigned int i = 0; i < reqFlavor.size(); i++)
	{
		if (reqFlavor[i] > 500)
			reqFlavor[i] = (int)(reqFlavor[i] * 0.5);
		else if (reqFlavor[i] > 400) // 该区段没有影响，400-500之间无数值
			reqFlavor[i] = (int)(reqFlavor[i] * 0.55);
		else if (reqFlavor[i] > 300) // 该区段没有影响，300-400之间无数值
			reqFlavor[i] = (int)(reqFlavor[i] * 0.55);
		else if (reqFlavor[i] > 200) // 该区段没有影响，200-300之间无数值
			reqFlavor[i] = (int)(reqFlavor[i] * 0.45);
		else if (reqFlavor[i] > 90)
			reqFlavor[i] = (int)(reqFlavor[i] * 0.5);
		else if (reqFlavor[i] > 50)
			reqFlavor[i] = (int)(reqFlavor[i] * 2.35);
		else
			reqFlavor[i] = (int)(reqFlavor[i] * 2.2);
	}
	saveResult(test, pred, reqFlavor, filename);
}

// time regular 1
void timeRegular1(TestInfo test, TrainInfo train, PredInfo pred, char *filename)
{

	vector<int> reqFlavor;
	vector<int> reqFlavor2;
	vector<int> reqFlavor3;
	vector<TrainFlavor> trainFlavorX;

	for (int i = 0; i < test.predictNum; i++)
	{
		// reqFlavor.push_back(63);
		TrainFlavor tmp;
		tmp.flavorX = pred.predFlavor[i];
		for (unsigned int j = 0; j < train.flavor.size(); j++)
		{
			if (train.flavor[j] == pred.predFlavor[i])
			{
				tmp.histDatetime.push_back(train.date[j]);
			}
		}
		trainFlavorX.push_back(tmp);
	}

	int n = train.day.size();
	int during = train.day[n - 1] - train.day[0] + 1; // days of the train data set

	for (int k = 0; k < test.predictNum; k++)
	{
		for (int i = 0, j = 0; (i < during) && (j < n); i++)
		{
			int cnt = 0;
			if ((train.day[0] + i) == train.day[j])
			{
				cnt = count(trainFlavorX[k].histDatetime.begin(), trainFlavorX[k].histDatetime.end(), train.day[j++]);
				trainFlavorX[k].histData.push_back(cnt);
			}
			else
				trainFlavorX[k].histData.push_back(0);
		}
	}

	// 濾波處理，主要是將極大值點降低一些
	for (int k = 0; k < test.predictNum; k++)
	{
		double sum = accumulate(trainFlavorX[k].histData.begin(), trainFlavorX[k].histData.end(), 0.0);
		double mean = sum / trainFlavorX[k].histData.size();
		double sq_sum = inner_product(trainFlavorX[k].histData.begin(), trainFlavorX[k].histData.end(), trainFlavorX[k].histData.begin(), 0.0);
		double stdev = sqrt(sq_sum / trainFlavorX[k].histData.size() - mean * mean);

		for (unsigned int i = 0; i < trainFlavorX[k].histData.size(); i++)
		{
			if (abs(trainFlavorX[k].histData[i] - mean) / (stdev) > 2.0)
			{
				// trainFlavorX[k].histData[i] = (int)(trainFlavorX[k].histData[i] * 0.65); //(int)(mean + 5); // 把極大值降低一些
			}
		}
	}

	for (int i = 0; i < test.predictNum; i++)
	{
		float tmp = 0;
		for (int j = 0; j < test.deltaDate; j++)
		{
			tmp += trainFlavorX[i].histData[during - 1 - j];
		}
		reqFlavor.push_back((int)(tmp));
	}

	for (int i = 0; i < test.predictNum; i++)
	{
		int tmp = 0;
		for (int j = 0; j < test.deltaDate; j++)
		{
			tmp += trainFlavorX[i].histData[during - 2 - j];
		}
		reqFlavor2.push_back(tmp);
	}
	for (int i = 0; i < test.predictNum; i++)
	{
		int tmp = 0;
		for (int j = 0; j < test.deltaDate; j++)
		{
			tmp += trainFlavorX[i].histData[during - 3 - j];
		}
		reqFlavor3.push_back(tmp);
	}
	for (unsigned int i = 0; i < reqFlavor.size(); i++)
	{
		reqFlavor[i] = (int)(reqFlavor[i] * 0.6 + reqFlavor2[i] * 0.3 + reqFlavor3[i] * 0.2); //* 1.3
	}

	saveResult(test, pred, reqFlavor, filename);
}

// time regular 0 以N为窗口，进行均值，与最后一个周期进行融合，
void timeRegular0(TestInfo test, TrainInfo train, PredInfo pred, char *filename)
{
	vector<int> reqFlavor;
	vector<TrainFlavor> trainFlavorX;
	// 提取预测虚拟机相关历史信息
	for (int i = 0; i < test.predictNum; i++)
	{
		TrainFlavor tmp;
		tmp.flavorX = pred.predFlavor[i];
		for (unsigned int j = 0; j < train.flavor.size(); j++)
		{
			if (train.flavor[j] == pred.predFlavor[i])
			{
				tmp.histDatetime.push_back(train.date[j]);
			}
		}
		trainFlavorX.push_back(tmp);
	}
	// 确实值填充为0
	int n = train.day.size();
	int during = train.day[n - 1] - train.day[0] + 1; // days of the train data set

	for (int k = 0; k < test.predictNum; k++)
	{
		for (int i = 0, j = 0; (i < during) && (j < n); i++)
		{
			int cnt = 0;
			if ((train.day[0] + i) == train.day[j])
			{
				cnt = count(trainFlavorX[k].histDatetime.begin(), trainFlavorX[k].histDatetime.end(), train.day[j++]);
				trainFlavorX[k].histData.push_back(cnt);
			}
			else
				trainFlavorX[k].histData.push_back(0); // 确实值填充为0
		}
	}

	// // 濾波處理，主要是將極大值點降低一些
	// for (int k = 0; k < test.predictNum; k++)
	// {
	// 	double sum = accumulate(trainFlavorX[k].histData.begin(), trainFlavorX[k].histData.end(), 0.0);
	// 	double mean = sum / trainFlavorX[k].histData.size();
	// 	double sq_sum = inner_product(trainFlavorX[k].histData.begin(), trainFlavorX[k].histData.end(), trainFlavorX[k].histData.begin(), 0.0);
	// 	double stdev = sqrt(sq_sum / trainFlavorX[k].histData.size() - mean * mean);

	// 	for (unsigned int i = 0; i < trainFlavorX[k].histData.size(); i++)
	// 	{
	// 		if (abs(trainFlavorX[k].histData[i] - mean) / (stdev) > 4.0)
	// 		{
	// 			 trainFlavorX[k].histData[i] = (int)(trainFlavorX[k].histData[i] * 0.95); //(int)(mean + 5); // 把極大值降低一些
	// 		}
	// 	}
	// }
	// 从最后要一天往前进行滑窗，以待预测天数N为窗口，
	for (int i = 0; i < test.predictNum; i++)
	{
		auto beginIter = trainFlavorX[i].histData.begin();
		auto endIter = trainFlavorX[i].histData.begin() + test.deltaDate;
		// auto endIter = trainFlavorX[i].histData.end();					  // 最后一天指针
		// auto beginIter = trainFlavorX[i].histData.end() - test.deltaDate; // 窗口的第一天指针

		std::vector<float> originalY;
		while (endIter <= trainFlavorX[i].histData.end())
		{
			double tmp = std::accumulate(beginIter, endIter, 0.0);
			originalY.push_back(tmp);

			endIter = endIter + 1;
			beginIter = beginIter + 1;
			// endIter = beginIter - 1;
			// beginIter = endIter - test.deltaDate;
		}
		// 待预测天数
		for (int k = 0; k < test.deltaDate + 3; k++)
		{
			float meanPred = std::accumulate(originalY.begin(), originalY.end(), 0.0) / originalY.size();
			float val = meanPred * 0.8 + originalY.back() * 0.6; //

			originalY.push_back((int)(val));
		}

		if (test.deltaDate < 7)
			reqFlavor.push_back((int)(originalY.back()));
		else
			reqFlavor.push_back((int)(originalY.back() + 5));
	}

	saveResult(test, pred, reqFlavor, filename);
}

//你要完成的功能总入口
void predict_server(char *info[MAX_INFO_NUM], char *data[MAX_DATA_NUM], int data_num, int info_num, char *filename)
{
	vector<string> res, res0; // split temp valirate

	//#############################Test Info####################################

	TestInfo testInfo; // save the info of test data set
	PredInfo predictInfo;

	// the first line
	testInfo.serverNum = stoi(info[0]); //服务器数量
	// 随后是不同服务器的CPU&MEM
	for (int i = 0; i < testInfo.serverNum; i++)
	{
		split(info[i + 1], ' ', res);
		testInfo.serverName.push_back(res[0]); // name of the server
		testInfo.CPU.push_back(stoi(res[1]));  // CPU of the server
		testInfo.MEM.push_back(stoi(res[2]));  // MEM of the server
	}

	// 读取虚拟机
	testInfo.predictNum = stoi(info[testInfo.serverNum + 2]);
	// xuniji
	for (int i = testInfo.serverNum + 3; i < testInfo.serverNum + 3 + testInfo.predictNum; i++)
	{
		split(info[i], ' ', res);
		predictInfo.predCPU.push_back(stoi(res[1]));		// cpu number
		predictInfo.predMEM.push_back(stoi(res[2]) / 1024); // mem number

		split(res[0], 'r', res0);
		predictInfo.predFlavor.push_back(stoi(res0[1])); // flavor number
	}

	// the first day
	split(info[testInfo.serverNum + 4 + testInfo.predictNum], ' ', res);
	string strFirstDay = res[0];
	testInfo.firstDay = g(res[0]);
	// the end day
	split(info[testInfo.serverNum + 5 + testInfo.predictNum], ' ', res);
	split(res[1], ':', res0);
	if (stoi(res0[0]))
	{
		testInfo.endDay = g(res[0]) + 1;
		testInfo.deltaDate = testInfo.endDay - testInfo.firstDay;
	}
	else
	{
		testInfo.endDay = g(res[0]);
		testInfo.deltaDate = testInfo.endDay - testInfo.firstDay;
	}

	testInfo.dayOfWeek = productDateVector(strFirstDay, testInfo.deltaDate);

	//#############################Test Info####################################

	//#############################Train Info###################################

	TrainInfo trainInfo; // save the info of  train data set

	for (int i = 0; i < data_num; i++)
	{
		split(data[i], '\t', res);
		split(res[1], 'r', res0);
		trainInfo.flavor.push_back(stoi(res0[1]));

		split(res[2], ' ', res0);
		trainInfo.date2.push_back(res0[0]);
		trainInfo.date.push_back(g(res0[0]));
		//trainInfo.dayOfWeek.push_back(res0[0]); // 曜日
	}

	trainInfo.day.assign(trainInfo.date.begin(), trainInfo.date.end()); // copy
	sort(trainInfo.day.begin(), trainInfo.day.end());					//sort
	trainInfo.day.erase(unique(trainInfo.day.begin(), trainInfo.day.end()), trainInfo.day.end());

	trainInfo.day2.assign(trainInfo.date2.begin(), trainInfo.date2.end()); // copy
	sort(trainInfo.day2.begin(), trainInfo.day2.end());					   //sort
	trainInfo.day2.erase(unique(trainInfo.day2.begin(), trainInfo.day2.end()), trainInfo.day2.end());

	trainInfo.dayOfWeek = productDateVector(trainInfo.date2[0], trainInfo.day.back() - trainInfo.day[0]);

	//#############################Train Info###################################

	//############################# Predict ###################################
	timeRegular0(testInfo, trainInfo, predictInfo, filename);
	//############################# Predict ###################################
}
