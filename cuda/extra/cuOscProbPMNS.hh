#pragma once

template<typename T>
void cuCalcComponent(T** xarg, T** xret, T** intern, T** params, unsigned int n, unsigned int m, T oscprobArgumentFactor, T DeltaMSq, T m_L);

template<typename T>
void cuCalcComponentCP(T** xarg, T** xret, T** intern, T** params, T m12, T m13, T m23, unsigned int n, unsigned int m, T oscprobArgumentFactor, T m_L);

template<typename T>
void cuCalcSum(T** xarg, T** xret, T w12, T w13, T w23, T wcp, bool isSame, unsigned int n);
