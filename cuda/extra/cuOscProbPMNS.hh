#pragma once

template<typename T>
void cuCalcComponent_modecos(T** xarg, T** xret, T** intern, T** params, unsigned int n, unsigned int m, T oscprobArgumentFactor, T DeltaMSq, T m_L);

template<typename T>
void cuCalcComponent_modesin(T** xarg, T** xret, T** intern, T** params, unsigned int n, unsigned int m, T oscprobArgumentFactor, T DeltaMSq, T m_L);

template<typename T>
void cuCalcComponentCP(T** xarg, T** xret, T** intern, T** params, T m12, T m13, T m23, unsigned int n, unsigned int m, T oscprobArgumentFactor, T m_L);

void cuCalcSum(double** xarg, double** xret, double w12, double w13, double w23, double wcp, bool isSame, unsigned int n);
