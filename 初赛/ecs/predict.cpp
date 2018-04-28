#include "predict.h"
// #include "ga.h"
#include "genneticAlgorithm.h"
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
	// CPU number of the server
	int CPU;
	// MEM number of the server
	int MEM;
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
std::vector<ResultNode> spSolution(int maxCPU, int maxMEM, int isCPU, std::vector<FlavorX> gaFlavorX, vector<int> &reqFlavor)
{

	std::vector<ResultNode> resFirst = knapsackFit(maxCPU, maxMEM, isCPU, gaFlavorX);

	// 修改每个预测值，先每个减1试试，看能不能减少一台物理服务器
	for (int j = 1; j < 3; j++)
	{
		for (unsigned int i = 0; i < gaFlavorX.size(); i++)
		{
			if (gaFlavorX[i].totalNum > j)
			{
				gaFlavorX[i].totalNum -= j;
				reqFlavor[i] -= j;
				std::vector<ResultNode> resTmp = knapsackFit(maxCPU, maxMEM, isCPU, gaFlavorX);
				if (resTmp.size() < resFirst.size())
				{
					return resTmp;
				}
				else // 否则还是恢复原来的预测值
				{
					gaFlavorX[i].totalNum += j;
					reqFlavor[i] += j;
				}
			}
		}
	}

	// 既然上面的方法无法降低分，那说明最后一台服务器是快要装满了的
	for (unsigned int j = 0; j < resFirst.size(); j++)
	{
		while ((resFirst[j].remMEM >= gaFlavorX[0].mem) && (resFirst[j].remCPU >= gaFlavorX[0].cpu))
		{
			for (unsigned int k = 0; k < gaFlavorX.size(); k++)
			{
				if ((resFirst[j].remMEM >= gaFlavorX[k].mem) && (resFirst[j].remCPU >= gaFlavorX[k].cpu))
				{
					resFirst[j].remMEM -= gaFlavorX[k].mem;
					resFirst[j].remCPU -= gaFlavorX[k].cpu;

					resFirst[j].flavor.push_back(gaFlavorX[k].flavor);
					resFirst[j].cpuNumber.push_back(gaFlavorX[k].cpu);
					resFirst[j].memNumber.push_back(gaFlavorX[k].mem);

					gaFlavorX[k].totalNum += 1;
					reqFlavor[k] += 1;
				}
			}
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

	// // 骚操作2 对内存大的预测值进行降低操作
	// if (!test.isCPU)
	// {
	// 	for (int i = 0; i < test.predictNum; i++)
	// 	{
	// 		if (pred.predMEM[i] / pred.predCPU[i] >= 4)
	// 		{
	// 			reqFlavor[i] = (int)(reqFlavor[i] * 0.85);
	// 		}
	// 	}
	// }

	for (int i = 0; i < test.predictNum; i++)
	{
		FlavorX tmp;
		tmp.flavor = pred.predFlavor[i];
		tmp.cpu = pred.predCPU[i];
		tmp.mem = pred.predMEM[i];
		tmp.totalNum = reqFlavor[i];
		gaTotalNum += reqFlavor[i];
		gaFlavor.push_back(tmp);

		for (int k = 0; k < reqFlavor[i]; k++)
		{
			needFlavor.push_back(pred.predFlavor[i]);
			needCPU.push_back(pred.predCPU[i]);
			needMEM.push_back(pred.predMEM[i]);
		}
	}

	std::vector<ResultNode> result = spSolution(test.CPU, test.MEM, test.isCPU, gaFlavor, reqFlavor);
	int serverNum = result.size();

	// std::vector<ResultNode> result = knapsackFit(test.CPU, test.MEM, test.isCPU, gaFlavor);
	// int serverNum = result.size();

	// vector<ResultNode> result = gaFit(50, 0.8, 0.5, test.CPU, test.MEM, test.isCPU, gaFlavor, gaTotalNum);
	// int serverNum = result.size();

	// // float ratioCPU = 1.0 - result[serverNum - 1].remCPU * 1.0 / test.CPU, ratioMEM = 1.0 - result[serverNum - 1].remMEM * 1.0 / test.MEM;

	// int cnt = min(result[serverNum - 1].remCPU / needCPU[0], result[serverNum - 1].remMEM / needMEM[0]);

	// for (int i = 0; i < test.predictNum; i++)
	// {
	// 	if (pred.predFlavor[i] == needFlavor[0])
	// 	{
	// 		reqFlavor[i] += cnt;
	// 	}
	// }

	// gaFlavor.clear();
	// gaTotalNum = 0;
	// for (int i = 0; i < test.predictNum; i++)
	// {
	// 	FlavorX tmp;
	// 	tmp.flavor = pred.predFlavor[i];
	// 	tmp.cpu = pred.predCPU[i];
	// 	tmp.mem = pred.predMEM[i];
	// 	tmp.totalNum = reqFlavor[i];
	// 	gaTotalNum += reqFlavor[i];
	// 	gaFlavor.push_back(tmp);
	// }

	// result = gaFit(50, 0.8, 0.5, test.CPU, test.MEM, test.isCPU, gaFlavor, gaTotalNum);
	// serverNum = result.size();

	// predict total flavor number
	int totalFlavor = accumulate(reqFlavor.begin(), reqFlavor.end(), 0);
	// result string
	string outputStr = to_string(totalFlavor) + "\n";
	for (int i = 0; i < test.predictNum; i++)
	{
		outputStr += "flavor" + to_string(pred.predFlavor[i]) + " " + to_string(reqFlavor[i]) + "\n";
	}

	outputStr += "\n" + to_string(serverNum);

	for (int i = 0; i < serverNum; i++)
	{
		vector<int> flavorTmp;
		flavorTmp.assign(result[i].flavor.begin(), result[i].flavor.end());
		sort(flavorTmp.begin(), flavorTmp.end());
		flavorTmp.erase(unique(flavorTmp.begin(), flavorTmp.end()), flavorTmp.end());

		outputStr += "\n" + to_string(i + 1) + " "; //printf("%d ", i);
		for (unsigned int j = 0; j < flavorTmp.size(); j++)
		{
			int cnt = count(result[i].flavor.begin(), result[i].flavor.end(), flavorTmp[j]);
			outputStr += "flavor" + to_string(flavorTmp[j]) + " " + to_string(cnt) + " "; //printf("flavor%d %d ", flavorTmp[j], cnt);
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

		float alpha = 0.63, S0 = 0;
		S0 = (originalY[0] + originalY[1]) / 2.0;
		S.push_back(S0);
		for (unsigned int k = 0; k < originalY.size(); k++)
		{
			double tmp = S[k] * (1 - alpha) + originalY[k]; // 正确的公式是 S(t+1) = (1-alpha)*S(t) + alpha * Y(t)
			S.push_back(tmp);
		}

		float val = originalY.back() * alpha + (1 - alpha) * S.back(); //; //alpha *
		reqFlavor.push_back((int)(val));
	}

	saveResult(test, pred, reqFlavor, filename);
}

// time regular 3
void timeRegular3(TestInfo test, TrainInfo train, PredInfo pred, char *filename)
{
	// flavor of train
	vector<int> flavor;
	// date of train
	vector<long> date;
	vector<string> date2;

	// 對flavor1-flavor15規則求和，提取總周期因子
	for (unsigned int i = 0; i < train.flavor.size(); i++)
	{ // delete the flavorX > 15
		if (train.flavor[i] < 16)
		{
			flavor.push_back(train.flavor[i]);
			date.push_back(train.date[i]);
			date2.push_back(train.date2[i]);
		}
	}
	train.flavor = flavor;
	train.date = date;
	train.date2 = date2;

	int n = train.day.size();
	int during = train.day[n - 1] - train.day[0] + 1; // days of the train data set

	TrainFlavor histTrain;

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
		if (abs(histTrain.histData[i] - mean0) / (stdev0) > 2.0)
		{
			// histTrain.histData[i] = (int)(histTrain.histData[i] * 0.65);
		}
	}

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
		medianFactor[i].mixFactor = medianTmp * 0.55 + meanTmp * 0.45;

		// printf("%.2f\n", medianTmp);
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
		reqFlavor.push_back((int)(res));
	}
	/*
		record:
		1. median:mean = 6:4 5days 76.18
		2. median:mean = 6:4 7days 76.52
		3. median:mean = 6:4 10days 77.96
		4. median:mean = 6:4 14days 72.8
	*/

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

	for (unsigned int i = 0; i < reqFlavor.size(); i++)
	{
		reqFlavor[i] = (int)(reqFlavor[i] * 0.36 + reqFlavor2[i] * 0.7); // * 0.35 + reqFlavor2[i] * 0.7 83.215 0.35:0.7
	}

	saveResult(test, pred, reqFlavor, filename);
}

// time regular 2
void timeRegular2(TestInfo test, TrainInfo train, PredInfo pred, char *filename)
{
	std::vector<int> reqFlavor;
	std::vector<TrainFlavor> trainFlavorX;

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
				float mean = sum / numList.size();

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
			medianFactor[i].mixFactor = max(medianTmp * 0.5 + meanTmp * 0.5, 0.01);
		}

		// last n days
		float base = 0;
		for (int j = 0; j < 10; j++)
		{
			int num = trainFlavorX[k].histData[during - 1 - j];
			int weekday = trainFlavorX[k].dayOfWeek[during - 1 - j];
			base = base + num / medianFactor[weekday - 1].mixFactor; // ;
		}
		base = base / 10;

		float res = 0;
		for (unsigned int i = 0; i < test.dayOfWeek.size(); i++)
		{
			res = res + base * medianFactor[test.dayOfWeek[i] - 1].mixFactor;
		}

		reqFlavor.push_back((int)(res));
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
		int tmp = 0;
		for (int j = 0; j < test.deltaDate; j++)
		{
			tmp += trainFlavorX[i].histData[during - 1 - j];
		}
		reqFlavor.push_back(tmp);
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
		reqFlavor[i] = (int)(reqFlavor[i] * 0.6 + reqFlavor2[i] * 0.3 + reqFlavor3[i] * 0.2); //
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
	split(info[0], ' ', res);	// get the info from the server
	testInfo.CPU = stoi(res[0]); // CPU of the server
	testInfo.MEM = stoi(res[1]); // MEM of the server
	// 2 line
	testInfo.predictNum = stoi(info[2]); // need to predict number of flavor
	// -4 line
	split(info[testInfo.predictNum + 4], '\n', res);
	testInfo.isCPU = res[0][0] == 'C' ? 0 : 1; // cpu, 0; mem, 1;
	// -2 line
	split(info[testInfo.predictNum + 6], ' ', res); // the first day
	string strFirstDay = res[0];
	testInfo.firstDay = g(res[0]);
	// -1 line
	split(info[testInfo.predictNum + 7], ' ', res); // the end day
	testInfo.endDay = g(res[0]);
	testInfo.deltaDate = testInfo.endDay - testInfo.firstDay;

	for (int i = 3; i < 3 + testInfo.predictNum; i++)
	{
		split(info[i], ' ', res);
		predictInfo.predCPU.push_back(stoi(res[1]));		// cpu number
		predictInfo.predMEM.push_back(stoi(res[2]) / 1024); // mem number

		split(res[0], 'r', res0);
		predictInfo.predFlavor.push_back(stoi(res0[1])); // flavor number
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
	timeRegular3(testInfo, trainInfo, predictInfo, filename);
	//############################# Predict ###################################

	// 需要输出的内容
	//char *result_file = (char *)"17\n\n0 8 0 20";

	// 直接调用输出文件的方法输出到指定文件中(ps请注意格式的正确性，如果有解，第一行只有一个数据；第二行为空；第三行开始才是具体的数据，数据之间用一个空格分隔开)
	//write_result(result_file, filename);
}
