#pragma once

template<typename T>
void cuCalcComponent(T** xarg, T** xret, T** intern, T** params, unsigned int n, unsigned int m, T oscprobArgumentFactor, T DeltaMSq, T m_L);

void cuCalcComponentCP(double** xarg, double** xret, double** intern, double** params, double m12, double m13, double m23, unsigned int n, unsigned int m, double oscprobArgumentFactor, double m_L);
void cuCalcSum(double** xarg, double** xret, double w12, double w13, double w23, double wcp, bool isSame, unsigned int n);
