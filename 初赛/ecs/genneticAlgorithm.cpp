// #include "ga.h"
#include "genneticAlgorithm.h"
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>
#include <sstream>

#define GA_MAX 500 // 遗传算法最大迭代次数

#define MAXN 5000
#define MAXC 10000

int d[MAXN][MAXC];

// 产生0-1之间的随机数，但是还没有确定是否好使
static float productRandNumber()
{
    return float(rand()) / float(RAND_MAX);
}

// 遗传算法中，种群的个体结构体
struct Person
{
    Person(int flavorxNum) : score(0.0), sumCPU(0), sumMEM(0), personID(0)
    {
        initGene(flavorxNum);
    }
    Person() : score(0.0), sumCPU(0), sumMEM(0), personID(0)
    {
        gene.clear();
    }

    Person &operator=(const Person &tmp)
    {
        score = tmp.score;
        sumCPU = tmp.sumCPU;
        sumMEM = tmp.sumMEM;
        personID = tmp.personID;
        gene = tmp.gene;
        return *this;
    }

    float score;           // 个体适应度
    int sumCPU;            // 个体cpu和
    int sumMEM;            // 个体mem和
    int personID;          // 个体在种群中的id
    std::vector<int> gene; // 个体的基因

    void initGene(int flavorxNum)
    {
        gene.assign(flavorxNum, 0);
    }
};

class GenneticAlgorithm
{
  private:
    std::vector<Person> gaPeople;   // 种群
    std::vector<FlavorX> gaFlavorX; // 待分配的虚拟机集合
    Person theBestPerson;           //最优个体
    int gaPeopleNum;                // 群体数量
    int gaTotalNum;                 // 虚拟机总数
    float gaCrossRate;              // 交叉概率
    float gaVariaRate;              // 变异概率

    int MAX_CPU; // 服务器最大cpu
    int MAX_MEM; //服务器最大mem
    int IS_CPU;  // 优化cpu还是mem

  public:
    static float totalScore(const std::vector<Person> &people)
    {
        float total = 0.0;
        for (unsigned int i = 0; i < people.size(); i++)
        {
            total += people[i].score;
        }
        return total;
    }
    static bool scoreSort(const Person &tmp1, const Person &tmp2)
    {
        return (tmp1.score <= tmp2.score);
    }

  public:
    GenneticAlgorithm(int peopleNum, float crossRate, float variaRate, int maxCPU, int maxMEM, int isCPU, std::vector<FlavorX> flavorX, int totalNum)
    {
        gaPeopleNum = peopleNum;
        gaCrossRate = crossRate;
        gaVariaRate = variaRate;
        gaFlavorX = flavorX;
        MAX_CPU = maxCPU;
        MAX_MEM = maxMEM;
        IS_CPU = isCPU;
        gaTotalNum = totalNum;
    }
    void findTheBestGene()
    {
        // 初始化种群
        initPeople();
        // 迭代遍历
        for (int i = 0; i < GA_MAX; i++)
        {
            // 计算适应度
            getScore();
            // 保存最优秀的基因
            saveTheBestPerson(i);
            // 自然选择 10%最优留下，40%轮盘选择留下 淘汰掉适应度较低的个体
            selectPerson();
            // 相邻个体交叉
            geneCross();
            // 隔5代发生变异
            if (i % 5 == 0 && i != 0)
            {
                geneVariation();
            }
        }
    }
    // 显示最优个体
    std::vector<int> bestResult()
    {
        return theBestPerson.gene;
    }

  private:
    // 初始化种群
    void initPeople()
    {
        // 依次初始化种群中的每一个个体
        int count = 0, remCPU = 0, remMEM = 0;
        for (int i = 0; i < gaPeopleNum; i++)
        {

            gaPeople.push_back(Person(gaTotalNum));
            Person &person = gaPeople.back();
            float memCapacity = ((rand() % 50 + 50) / 100.0) * MAX_MEM, cpuCapacity = ((rand() % 50 + 50) / 100.0) * MAX_CPU;
            count = 0;
            remCPU = 0;
            remMEM = 0;

            // 初始化个体的基因
            while ((remCPU <= cpuCapacity) && (remMEM <= memCapacity))
            {
                int idx = rand() % gaTotalNum;
                if (count == 3)
                    break;
                if (person.gene[idx])
                {
                    count++;
                    continue;
                }
                person.gene[idx] = 1;
                remCPU += gaFlavorX[idx].cpu;
                remMEM += gaFlavorX[idx].mem;
            }
        }
        // 初始化最优个体
        theBestPerson.initGene(gaTotalNum);
    }

    // 保存最优秀的个体基因
    void saveTheBestPerson(int personId)
    {
        // 对种群内所有个体进行排序，适应度最高的在最后
        std::stable_sort(gaPeople.begin(), gaPeople.end(), GenneticAlgorithm::scoreSort);
        // 取适应度最高的个体
        Person &person = gaPeople.back();
        // 保存最优个体
        if (person.score > theBestPerson.score)
        {
            theBestPerson = person;
            theBestPerson.personID = personId;
        }
    }

    // 计算个体适应度
    void getScore()
    {
        for (unsigned int i = 0; i < gaPeople.size(); i++)
        { // 取种群中某个个体的基因，计算其适应度
            Person &person = gaPeople[i];
            int sumMEM = 0, sumCPU = 0;
            // 遍历该基因的各个染色体，1表示装箱，0表示不装箱
            for (unsigned int j = 0; j < person.gene.size(); j++)
            {
                if (person.gene[j])
                { // 总内存用量
                    sumMEM += gaFlavorX[j].mem;
                    // 总CPU用量
                    sumCPU += gaFlavorX[j].cpu;
                }
            }
            // 如果CPU&MEM资源超过物理服务器最大资源，则其适应度最低,为0; 分越高越好
            if ((sumMEM > MAX_MEM) || (sumCPU > MAX_CPU))
            {
                person.score = 0;
                continue;
            }
            // 0，CPU；1,MEM；
            if (IS_CPU)
            {
                person.score = sumMEM * 100.0 / MAX_MEM; // 该个体的适应度等于內存利用率+  sumCPU * 1.0 / MAX_CPU
            }
            else
            {
                person.score = sumCPU * 100.0 / MAX_CPU; // 该个体的适应度等于CPU利用率 +  sumMEM * 1.0 / MAX_MEM
            }

            person.sumCPU = sumCPU; // 在同利用率的情況下，選擇sumCpu大的
            person.sumMEM = sumMEM; // 在同利用率的情况下，选择内存大的
        }
    }

    // 自然选择
    void selectPerson()
    {
        std::vector<Person> newPeople;
        // 排序
        std::stable_sort(gaPeople.begin(), gaPeople.end(), GenneticAlgorithm::scoreSort);
        int oldPeopleNum = gaPeople.size();

        // 保留群体中最优的前10%
        int saveNum = (int)(oldPeopleNum * 0.1);
        for (int i = 0; i < saveNum; i++)
        {
            newPeople.push_back(gaPeople.back());
            gaPeople.pop_back();
        }

        // 计算剩余个体的累积概率
        std::vector<float> selectedRate;
        float sumScore = GenneticAlgorithm::totalScore(gaPeople);
        selectedRate.push_back(gaPeople[0].score / sumScore);
        for (unsigned int i = 1; i < gaPeople.size(); i++)
        {
            float cur_rate = selectedRate.back() + (gaPeople[i].score / sumScore);
            selectedRate.push_back(cur_rate);
        }

        // 利用轮赌法选择剩下的40%
        int left_num = (int)(oldPeopleNum * 0.4);
        for (int i = 0; i < left_num; i++)
        {
            float rand_rate = productRandNumber();
            for (unsigned int idx = 0; idx < selectedRate.size(); idx++)
            {
                if (rand_rate <= selectedRate[idx])
                {
                    newPeople.push_back(gaPeople[idx]);
                    break;
                }
            }
        }

        // 新群体赋值
        gaPeople.clear();
        gaPeople = newPeople;
    }

    // 是否交叉
    bool isCross()
    {
        return (productRandNumber() <= gaCrossRate);
    }

    // 交叉
    void geneCross()
    {
        int peopleNum = gaPeople.size();
        // 相邻个体间进行交叉
        for (int i = 0; i < peopleNum - 1; i += 2)
        {
            Person tmp1 = gaPeople[i];
            Person tmp2 = gaPeople[i + 1];
            // 取相邻两个个体，看看是否交叉
            for (unsigned int j = 0; j < tmp1.gene.size(); j++)
            {
                if (isCross())
                {
                    int tmp = tmp1.gene[j];
                    tmp1.gene[j] = tmp2.gene[j];
                    tmp2.gene[j] = tmp;
                }
            }
            // 交叉后的新个体，放入种群
            gaPeople.push_back(tmp1);
            gaPeople.push_back(tmp2);
        }
    }
    // 是否变异
    bool isVariation()
    {
        return (productRandNumber() <= gaVariaRate);
    }

    // 变异
    void geneVariation()
    {
        for (unsigned int i = 0; i < gaPeople.size(); i++)
        {
            if (isVariation())
            {
                Person &person = gaPeople[i];
                for (unsigned int j = 0; j < person.gene.size(); j++)
                { // 每一位都有一定概率变异
                    if (isVariation())
                    {
                        person.gene[j] = (person.gene[j] ? 0 : 1);
                    }
                }
            }
        }
    }
};

// 遗传算法贪心多背包
std::vector<ResultNode> gaFit(int peopleNum, float crossRate, float variteRate, int maxCPU, int maxMEM, int isCPU, std::vector<FlavorX> flavorx, int totalNum)
{
    std::vector<ResultNode> resNodeList;

    while (totalNum)
    {
        std::vector<FlavorX> tmp;
        int cnt = 0;
        for (unsigned int i = 0; i < flavorx.size(); i++)
        {
            if (flavorx[i].totalNum)
            {
                for (int j = 0; j < flavorx[i].totalNum; j++)
                {
                    tmp.push_back(flavorx[i]);
                    cnt++;
                }
            }
        }

        GenneticAlgorithm ga(peopleNum, crossRate, variteRate, maxCPU, maxMEM, isCPU, tmp, cnt);
        ga.findTheBestGene();
        std::vector<int> res = ga.bestResult();
        totalNum -= std::accumulate(res.begin(), res.end(), 0);

        ResultNode resNode;
        resNode.remMEM = maxMEM;
        resNode.remCPU = maxCPU;
        for (unsigned int i = 0; i < res.size(); i++)
        {
            if (res[i])
            {
                resNode.flavor.push_back(tmp[i].flavor);
                resNode.cpuNumber.push_back(tmp[i].cpu);
                resNode.memNumber.push_back(tmp[i].mem);
                resNode.remCPU -= tmp[i].cpu;
                resNode.remMEM -= tmp[i].mem;

                for (unsigned int j = 0; j < flavorx.size(); j++)
                {
                    if (tmp[i].flavor == flavorx[j].flavor)
                    {
                        flavorx[j].totalNum -= 1;
                    }
                }
            }
        }
        resNodeList.push_back(resNode);
    }
    return resNodeList;
}

std::vector<ResultNode> knapsack_mem(int maxCPU, int maxMEM, std::vector<FlavorX> flavorx)
{
    // 物理服务器结点列表
    std::vector<ResultNode> resNodeList;
    // 状态矩阵基本单元
    D_unit dUnit;
    dUnit.remCPU = maxCPU;
    dUnit.remMEM = maxMEM;

    while (!flavorx.empty())
    {
        // 构建状态转移矩阵
        std::vector<std::vector<D_unit> > D;
        // 随机打乱
        // std::random_shuffle(flavorx.begin(), flavorx.end());
        // 初始化
        for (unsigned int i = 0; i < flavorx.size(); i++)
        {
            std::vector<D_unit> unit(maxMEM + 1, dUnit);
            D.push_back(unit);
        }

        // D[0][maxMEM - flavorx[0].mem] = dUnit;
        D[0][maxMEM - flavorx[0].mem].remCPU -= flavorx[0].cpu;
        D[0][maxMEM - flavorx[0].mem].remMEM -= flavorx[0].mem;

        int j = maxMEM;
        unsigned int i = 1;
        for (i = 1; i < flavorx.size(); i++)
        {
            for (j = maxMEM; j >= 0; j--)
            {
                D_unit unit_tmp = D[i - 1][j];

                if (j + flavorx[i].mem <= maxMEM)
                {
                    if (D[i - 1][j + flavorx[i].mem].remCPU - flavorx[i].cpu >= 0 &&
                        D[i - 1][j + flavorx[i].mem].remMEM - flavorx[i].mem >= 0 &&
                        D[i - 1][j + flavorx[i].mem].remMEM == j + flavorx[i].mem)
                    {
                        // 下一状态更小
                        if (D[i - 1][j + flavorx[i].mem].remMEM - flavorx[i].mem < D[i - 1][j].remMEM)
                        {
                            unit_tmp = D[i - 1][j + flavorx[i].mem];
                            unit_tmp.remCPU -= flavorx[i].cpu;
                            unit_tmp.remMEM -= flavorx[i].mem;
                        }
                    }
                    else
                    {
                        unit_tmp = D[i - 1][j];
                    }
                }
                D[i][j] = unit_tmp;
            }
            if (D[i][0].remMEM == 0)
                break;
        }

        if (i == flavorx.size())
            i = flavorx.size() - 1;
        for (j = 0; j <= maxMEM; j++)
        {
            if (D[i][j].remMEM == j)
                break;
        }

        ResultNode resNode;
        resNode.remMEM = maxMEM;
        resNode.remCPU = maxCPU;
        while (i > 0)
        { // 说明flaovrx[i]不未放入服务器
            if (D[i][j].remMEM == D[i - 1][j].remMEM)
            {
                i--;
            }
            else
            {
                resNode.flavor.push_back(flavorx[i].flavor);
                resNode.cpuNumber.push_back(flavorx[i].cpu);
                resNode.memNumber.push_back(flavorx[i].mem);

                resNode.remMEM -= flavorx[i].mem;
                resNode.remCPU -= flavorx[i].cpu;

                j += flavorx[i].mem;

                auto iter = flavorx.begin() + i;
                flavorx.erase(iter);

                i--;
            }
        }
        // 说明第1台虚拟机在服务器上， 将其删除
        if (D[0][j].remMEM == j && j != maxMEM)
        {
            resNode.flavor.push_back(flavorx[i].flavor);
            resNode.cpuNumber.push_back(flavorx[i].cpu);
            resNode.memNumber.push_back(flavorx[i].mem);

            resNode.remMEM -= flavorx[i].mem;
            resNode.remCPU -= flavorx[i].cpu;

            flavorx.erase(flavorx.begin());
        }

        if (!resNode.flavor.empty())
            resNodeList.push_back(resNode);
    }

    return resNodeList;
}

std::vector<ResultNode> knapsack_cpu(int maxCPU, int maxMEM, std::vector<FlavorX> flavorx)
{
    // 物理服务器结点列表
    std::vector<ResultNode> resNodeList;
    // 状态矩阵基本单元
    D_unit dUnit;
    dUnit.remCPU = maxCPU;
    dUnit.remMEM = maxMEM;

    while (!flavorx.empty())
    {
        // 构建状态转移矩阵
        std::vector<std::vector<D_unit> > D;
        // 随机打乱
        // std::random_shuffle(flavorx.begin(), flavorx.end());
        // 初始化
        for (unsigned int i = 0; i < flavorx.size(); i++)
        {
            std::vector<D_unit> unit(maxCPU + 1, dUnit);
            D.push_back(unit);
        }

        // D[0][maxMEM - flavorx[0].mem] = dUnit;
        D[0][maxCPU - flavorx[0].cpu].remCPU -= flavorx[0].cpu;
        D[0][maxCPU - flavorx[0].cpu].remMEM -= flavorx[0].mem;

        unsigned int i = 1;
        int j = maxCPU;
        for (i = 1; i < flavorx.size(); i++)
        {
            for (j = maxCPU; j >= 0; j--)
            {
                D_unit unit_tmp = D[i - 1][j];

                if (j + flavorx[i].cpu <= maxCPU)
                {
                    if (D[i - 1][j + flavorx[i].cpu].remCPU - flavorx[i].cpu >= 0 &&
                        D[i - 1][j + flavorx[i].cpu].remMEM - flavorx[i].mem >= 0 &&
                        D[i - 1][j + flavorx[i].cpu].remCPU == j + flavorx[i].cpu)
                    {
                        // 下一状态更小
                        if (D[i - 1][j + flavorx[i].cpu].remCPU - flavorx[i].cpu < D[i - 1][j].remCPU)
                        {
                            unit_tmp = D[i - 1][j + flavorx[i].cpu];
                            unit_tmp.remCPU -= flavorx[i].cpu;
                            unit_tmp.remMEM -= flavorx[i].mem;
                        }
                    }
                    else
                    {
                        unit_tmp = D[i - 1][j];
                    }
                }
                D[i][j] = unit_tmp;
            }
            if (D[i][0].remCPU == 0)
                break;
        }

        if (i == flavorx.size())
            i = flavorx.size() - 1;
        for (j = 0; j <= maxCPU; j++)
        {
            if (D[i][j].remCPU == j)
                break;
        }

        ResultNode resNode;
        resNode.remCPU = maxCPU;
        resNode.remMEM = maxMEM;

        while (i > 0)
        { // 说明flaovrx[i]未放入服务器
            if (D[i][j].remCPU == D[i - 1][j].remCPU)
            {
                i--;
            }
            else
            {
                resNode.flavor.push_back(flavorx[i].flavor);
                resNode.cpuNumber.push_back(flavorx[i].cpu);
                resNode.memNumber.push_back(flavorx[i].mem);

                resNode.remMEM -= flavorx[i].mem;
                resNode.remCPU -= flavorx[i].cpu;

                j += flavorx[i].cpu;
                auto iter = flavorx.begin() + i;
                flavorx.erase(iter);
                i--;
            }
        }
        // 说明第1台虚拟机在服务器上， 将其删除
        if (D[0][j].remCPU == j && j != maxCPU)
        {
            resNode.flavor.push_back(flavorx[i].flavor);
            resNode.cpuNumber.push_back(flavorx[i].cpu);
            resNode.memNumber.push_back(flavorx[i].mem);

            resNode.remMEM -= flavorx[i].mem;
            resNode.remCPU -= flavorx[i].cpu;

            flavorx.erase(flavorx.begin());
        }

        if (!resNode.flavor.empty())
            resNodeList.push_back(resNode);
    }

    return resNodeList;
}

std::vector<ResultNode> knapsackFit1(int maxCPU, int maxMEM, int isCPU, std::vector<FlavorX> flavorx)
{

    std::vector<FlavorX> tmp;
    int cnt = 0;
    for (unsigned int i = 0; i < flavorx.size(); i++)
    {
        if (flavorx[i].totalNum)
        {
            for (int j = 0; j < flavorx[i].totalNum; j++)
            {
                tmp.push_back(flavorx[i]);
                cnt++;
            }
        }
    }
    // 随机打乱
    std::random_shuffle(tmp.begin(), tmp.end());

    std::vector<ResultNode> resNodeList;
    if (isCPU)
    {
        resNodeList = knapsack_mem(maxCPU, maxMEM, tmp);
    }
    else
    {
        resNodeList = knapsack_mem(maxCPU, maxMEM, tmp);
    }
    return resNodeList;
}

std::vector<ResultNode> knapsackFit(int maxCPU, int maxMEM, int isCPU, std::vector<FlavorX> flavorx)
{

    int tmpNum;
    int serverNum = 1000;
    std::vector<ResultNode> tmpResult;
    std::vector<ResultNode> result;

    for (int i = 0; i < 50; i++)
    {
        tmpResult = knapsackFit1(maxCPU, maxMEM, isCPU, flavorx);
        tmpNum = tmpResult.size();

        if (tmpNum < serverNum)
        {
            serverNum = tmpNum;
            result = tmpResult;
        }
    }
    return result;
}