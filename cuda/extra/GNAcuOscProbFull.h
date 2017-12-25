#ifndef GNACUOSCPROBFULL_H
#define GNACUOSCPROBFULL_H
#include "GNAcuOscProbMem.hh"
#include "GNAcuRootMath.h"


extern "C" void calcCuFullProb_double(GNAcuOscProbMem<double>& mem,
				      double DMSq12, double DMSq13,
				      double DMSq23, double weight12,
				      double weight13, double weight23,
				      double weightCP, double* ret, double L,
				      double* Enu, int EnuSize, bool sameAB);

extern "C" void calcCuFullProb_float(GNAcuOscProbMem<float>& mem, float DMSq12,
				     float DMSq13, float DMSq23, float weight12,
				     float weight13, float weight23,
				     float weightCP, float* ret, float L,
				     float* Enu, int EnuSize, bool sameAB);

#endif /* GNACUOSCPROBFULL_H */
