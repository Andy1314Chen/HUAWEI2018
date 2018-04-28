#ifndef __GA__H_
#define __GA__H_
#include <vector>

// 物品定义
struct FlavorX
{
    int cpu;      // cpu
    int mem;      // mem
    int flavor;   // flavorX
    int totalNum; // need to predict
};

// 存放物理服務器信息
struct ResultNode
{
    // flavor
    std::vector<int> flavor;
    // cpu num
    std::vector<int> cpuNumber;
    // mem num
    std::vector<int> memNumber;
    // remCPU
    int remCPU;
    // remMEM
    int remMEM;
    // 利用率
    float radio;
};

struct D_unit
{
    int remCPU;
    int remMEM;
};


std::vector<ResultNode> gaFit(int gn, float cr, float vr, int maxCPU, int maxMEM, int isCPU, std::vector<FlavorX> flavorx, int totalNum);
std::vector<ResultNode> knapsackFit(int maxCPU, int maxMEM, int isCPU, std::vector<FlavorX> flavorx);

#endif