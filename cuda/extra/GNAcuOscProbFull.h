#ifndef GNACUOSCPROBFULL_H
#define GNACUOSCPROBFULL_H
#include "GNAcuOscProbMem.hh"


//extern template class GNAcuOscProbMem<double>(int in);
extern "C" void test (double* data);
extern "C" void calcCuFullProb (GNAcuOscProbMem<double> &mem,
  				double DMSq12, double DMSq13, double DMSq23,
                        	double weight12, double weight13, double weight23, double weightCP,
                        	double* ret, double L, double* Enu, int EnuSize, bool sameAB);
#endif /* GNACUOSCPROBFULL_H */
