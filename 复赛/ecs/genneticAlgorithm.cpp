// #include "ga.h"
#include "genneticAlgorithm.h"
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>
#include <sstream>

#define GA_MAX 300 // 遗传算法最大迭代次数

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
    Person(int flavorxNum) : score(5.0), sumCPU(0), sumMEM(0), personID(0)
    {
        initGene(flavorxNum);
    }
    Person() : score(5.0), sumCPU(0), sumMEM(0), personID(0)
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

    std::vector<int> MAX_CPU; // 服务器最大cpu
    std::vector<int> MAX_MEM; //服务器最大mem
    int serverNum;            // 优化cpu还是mem

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
    GenneticAlgorithm(int peopleNum, float crossRate, float variaRate, std::vector<int> maxCPU, std::vector<int> maxMEM, std::vector<FlavorX> flavorX, int totalNum)
    {
        gaPeopleNum = peopleNum;
        gaCrossRate = crossRate;
        gaVariaRate = variaRate;
        gaFlavorX = flavorX;
        MAX_CPU = maxCPU;
        MAX_MEM = maxMEM;
        serverNum = maxCPU.size();
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
        printf("best score:%.2f\n", theBestPerson.score);
        return theBestPerson.gene;
    }

  private:
    // 初始化种群
    void initPeople()
    {
        // 依次初始化种群中的每一个个体
        for (int i = 0; i < gaPeopleNum; i++)
        {

            gaPeople.push_back(Person(gaTotalNum));
            Person &person = gaPeople.back();

            // 初始化种群
            for (int m = 0; m < serverNum; m++)
            {
                float memCapacity = ((rand() % 50 + 50) / 100.0) * MAX_MEM[m], cpuCapacity = ((rand() % 50 + 50) / 100.0) * MAX_CPU[m];
                int count = 0, remCPU = 0, remMEM = 0;

                // 初始化个体的基因
                while ((remCPU <= cpuCapacity) && (remMEM <= memCapacity))
                {
                    int idx = rand() % gaTotalNum; // 随机选择第i个染色体
                    if (count == 5)
                        break;
                    if (person.gene[idx])
                    {
                        count++;
                        continue;
                    }
                    person.gene[idx] = rand() % (serverNum + 1); // 随机装入N(N=0,1,2,3)号箱子
                    if (person.gene[idx] == m)
                    {
                        remCPU += gaFlavorX[idx].cpu;
                        remMEM += gaFlavorX[idx].mem;
                    }
                }
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
            // printf("best score: %.2f\n", theBestPerson.score);
        }
    }

    // 计算个体适应度
    void getScore()
    {
        for (unsigned int i = 0; i < gaPeople.size(); i++)
        {
            // 取种群中某个个体的基因，计算其适应度
            Person &person = gaPeople[i];
            int flag = 0; // 资源超分标志

            std::vector<D_unit> totalCPUMEM; // 当前方案
            std::vector<int> isUse;          // 某个箱子是否使用了
            for (int m = 0; m < serverNum; m++)
            {
                D_unit tmp;
                tmp.remCPU = MAX_CPU[m];
                tmp.remMEM = MAX_MEM[m];
                totalCPUMEM.push_back(tmp);

                isUse.push_back(0);
            }
            // 遍历该个体的各个染色体，0表示不装箱，1表示1号箱，2表示2号箱，3表示3号箱
            for (unsigned int j = 0; j < person.gene.size(); j++)
            {
                if (person.gene[j]) // 计算服务器的剩余容量
                {
                    totalCPUMEM[person.gene[j] - 1].remCPU -= gaFlavorX[j].cpu;
                    totalCPUMEM[person.gene[j] - 1].remMEM -= gaFlavorX[j].mem;

                    if ((totalCPUMEM[person.gene[j] - 1].remCPU < 0) || (totalCPUMEM[person.gene[j] - 1].remMEM < 0))
                    {
                        flag = 1; //超分
                        person.score = 5;
                        break;
                    }

                    isUse[person.gene[j] - 1] = 1; // 如果使用了，则置1
                }
            }
            if (flag)
                continue;
            // 如果CPU&MEM资源超过物理服务器最大资源，则其适应度最低,为0; 分越高越好
            for (int k = 0; k < serverNum; k++)
            {
                if ((totalCPUMEM[k].remCPU < 0) || (totalCPUMEM[k].remMEM < 0))
                {
                    person.score = 5; // 如果设为0,则报错，不知道为什么？
                    flag = 1;
                    break;
                }
            }
            if (flag)
                continue;
            // 如果资源没有超分，则计算利用率作为适应度函数
            float f1_cpu = 0, f1_mem = 0, f2_cpu = 0, f2_mem = 0;
            int time = 0;
            for (int k = 0; k < serverNum; k++)
            {
                f1_cpu += totalCPUMEM[k].remCPU * isUse[k];
                f1_mem += totalCPUMEM[k].remMEM * isUse[k];
                f2_cpu += MAX_CPU[k] * isUse[k];
                f2_mem += MAX_MEM[k] * isUse[k];

                if (isUse[k])
                {
                    time++;
                }
            }
            // 总资源利用率作为适应度函数值
            person.score = (1 - f1_cpu / f2_cpu) * 50 + (1 - f1_mem / f2_mem) * 50 - time * 10;
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

            // 单点交叉算子
            // int idx = rand() % gaTotalNum; // 随机选择第i个染色体
            // while (idx == 0)               // 必须产生一个不为0的
            //     idx = rand() % gaTotalNum;
            // if (isCross()) // 看看概率是否要进行交叉
            // {
            //     for (unsigned int j = idx; j < tmp1.gene.size(); j++) // 某个基因后进行交叉
            //     {
            //         int tmp = tmp1.gene[j];
            //         tmp1.gene[j] = tmp2.gene[j];
            //         tmp2.gene[j] = tmp;
            //     }
            //     // 交叉后的两个新个体，放入种群中
            //     gaPeople.push_back(tmp1);
            //     gaPeople.push_back(tmp2);
            // }

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
                        // person.gene[j] = (person.gene[j] ? 0 : 1);
                        int tmp = rand() % (serverNum + 1); // 变异
                        while (tmp == person.gene[j])       // 一定要变异跟原来不一样
                            tmp = rand() % (serverNum + 1);
                        person.gene[j] = tmp;
                    }
                }
            }
        }
    }
};

// 遗传算法贪心多背包
std::vector<ResultNode> gaFit(int peopleNum, float crossRate, float variteRate, std::vector<int> maxCPU, std::vector<int> maxMEM, std::vector<FlavorX> flavorx, int totalNum)
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
        // 随机打乱
        std::random_shuffle(tmp.begin(), tmp.end());

        GenneticAlgorithm ga(peopleNum, crossRate, variteRate, maxCPU, maxMEM, tmp, cnt);
        ga.findTheBestGene();
        std::vector<int> res = ga.bestResult();
        // 解析刚刚放入服务器中的那一波虚拟机
        for (unsigned int i = 0; i < maxCPU.size(); i++)
        {
            ResultNode resNode;
            resNode.serverClass = i;
            resNode.remMEM = maxMEM[i];
            resNode.remCPU = maxCPU[i];

            for (unsigned int k = 0; k < res.size(); k++)
            {
                if (res[k] == i + 1) // 该虚拟机被装箱了,0 是没装；1 是装1号；2 是装2号；3 是装3号；
                {
                    totalNum--;

                    resNode.flavor.push_back(tmp[k].flavor);
                    resNode.cpuNumber.push_back(tmp[k].cpu);
                    resNode.memNumber.push_back(tmp[k].mem);
                    resNode.remCPU -= tmp[k].cpu;
                    resNode.remMEM -= tmp[k].mem;

                    if (resNode.remCPU < 0)
                    {
                        for (int q = 0; q < res.size(); q++)
                        {
                            if (res[q])
                                printf("flavor%d cpu:%d mem:%d res:%d \n", tmp[q].flavor, tmp[q].cpu, tmp[q].mem, res[q]);
                        }
                    }

                    for (unsigned int m = 0; m < flavorx.size(); m++)
                    {
                        if (tmp[k].flavor == flavorx[m].flavor)
                            flavorx[m].totalNum -= 1;
                    }
                }
            }
            if (resNode.flavor.empty())
                continue;
            resNodeList.push_back(resNode);
        }
    }
    return resNodeList;
}

std::vector<ResultNode> knapsack_mem(std::vector<int> MAXCPU, std::vector<int> MAXMEM, std::vector<FlavorX> flavorx)
{
    // 物理服务器结点列表
    std::vector<ResultNode> resNodeList;

    while (!flavorx.empty())
    {
        // int s = flavorx.size();
        // printf("size: %d\n", s);
        ResultNode bestResNode;
        bestResNode.radio = 0.0;
        std::vector<FlavorX> bestFlavorX = flavorx;
        // 对每个resNode都尝试进行多个不同的箱子，找到利用率最高的
        for (unsigned int m = 0; m < MAXCPU.size(); m++)
        {
            // 相同的虚拟机
            std::vector<FlavorX> tmpFlavorX = flavorx;
            int maxMEM = MAXMEM[m], maxCPU = MAXCPU[m];
            // 状态矩阵基本单元
            D_unit dUnit;
            dUnit.remCPU = maxCPU;
            dUnit.remMEM = maxMEM;
            // 构建状态转移矩阵
            std::vector<std::vector<D_unit> > D;
            for (unsigned int i = 0; i < tmpFlavorX.size(); i++)
            {
                std::vector<D_unit> unit(maxMEM + 1, dUnit);
                D.push_back(unit);
            }
            // D[0][maxMEM - flavorx[0].mem] = dUnit;
            D[0][maxMEM - tmpFlavorX[0].mem].remCPU -= tmpFlavorX[0].cpu;
            D[0][maxMEM - tmpFlavorX[0].mem].remMEM -= tmpFlavorX[0].mem;

            int j = maxMEM;
            unsigned int i = 1;
            for (i = 1; i < tmpFlavorX.size(); i++)
            {
                for (j = maxMEM; j >= 0; j--)
                {
                    D_unit unit_tmp = D[i - 1][j];

                    if (j + tmpFlavorX[i].mem <= maxMEM)
                    {
                        if (D[i - 1][j + tmpFlavorX[i].mem].remCPU - tmpFlavorX[i].cpu >= 0 &&
                            D[i - 1][j + tmpFlavorX[i].mem].remMEM - tmpFlavorX[i].mem >= 0 &&
                            D[i - 1][j + tmpFlavorX[i].mem].remMEM == j + tmpFlavorX[i].mem)
                        {
                            // 下一状态更小
                            if (D[i - 1][j + tmpFlavorX[i].mem].remMEM - tmpFlavorX[i].mem < D[i - 1][j].remMEM)
                            {
                                unit_tmp = D[i - 1][j + tmpFlavorX[i].mem];
                                unit_tmp.remCPU -= tmpFlavorX[i].cpu;
                                unit_tmp.remMEM -= tmpFlavorX[i].mem;
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
            if (i == tmpFlavorX.size())
                i = tmpFlavorX.size() - 1;
            for (j = 0; j <= maxMEM; j++)
            {
                if (D[i][j].remMEM == j)
                    break;
            }

            ResultNode resNode;
            resNode.remMEM = maxMEM;
            resNode.remCPU = maxCPU;
            resNode.serverClass = m;
            while (i > 0)
            { // 说明flaovrx[i]未放入服务器
                if (D[i][j].remMEM == D[i - 1][j].remMEM)
                {
                    i--;
                }
                else
                {
                    resNode.flavor.push_back(tmpFlavorX[i].flavor);
                    resNode.cpuNumber.push_back(tmpFlavorX[i].cpu);
                    resNode.memNumber.push_back(tmpFlavorX[i].mem);

                    resNode.remMEM -= tmpFlavorX[i].mem;
                    resNode.remCPU -= tmpFlavorX[i].cpu;

                    j += tmpFlavorX[i].mem;

                    auto iter = tmpFlavorX.begin() + i;
                    tmpFlavorX.erase(iter);

                    i--;
                }
            }
            // 说明第1台虚拟机在服务器上， 将其删除
            if (D[0][j].remMEM == j && j != maxMEM)
            {
                resNode.flavor.push_back(tmpFlavorX[i].flavor);
                resNode.cpuNumber.push_back(tmpFlavorX[i].cpu);
                resNode.memNumber.push_back(tmpFlavorX[i].mem);

                resNode.remMEM -= tmpFlavorX[i].mem;
                resNode.remCPU -= tmpFlavorX[i].cpu;

                tmpFlavorX.erase(tmpFlavorX.begin());
            }
            // 计算该箱子的利用率并保存相关信息
            resNode.radio = (1 - resNode.remMEM * 1.0 / MAXMEM[m]) * 50.0 + (1 - resNode.remCPU * 1.0 / MAXCPU[m]) * 50.0;
            // 保存最优信息
            if (resNode.radio > bestResNode.radio)
            {
                bestResNode = resNode;
                bestFlavorX = tmpFlavorX;
            }
        }
        if (!bestResNode.flavor.empty())
        {
            resNodeList.push_back(bestResNode);
            flavorx = bestFlavorX;
        }
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

std::vector<ResultNode> knapsackFit1(std::vector<int> maxCPU, std::vector<int> maxMEM, std::vector<FlavorX> flavorx)
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

    resNodeList = knapsack_mem(maxCPU, maxMEM, tmp);

    return resNodeList;
}

std::vector<ResultNode> knapsackFit(std::vector<int> maxCPU, std::vector<int> maxMEM, std::vector<FlavorX> flavorx)
{

    int tmpNum;
    int serverNum = 1000;
    std::vector<ResultNode> tmpResult;
    std::vector<ResultNode> result;

    for (int i = 0; i < 20; i++)
    {
        tmpResult = knapsackFit1(maxCPU, maxMEM, flavorx);
        tmpNum = tmpResult.size();

        if (tmpNum < serverNum)
        {
            serverNum = tmpNum;
            result = tmpResult;
        }
    }
    return result;
}