#ifndef GNACUOSCPROBMEM_H
#define GNACUOSCPROBMEM_H

#include <iostream>
#include "GNAcuRootMath.h"
enum GpuMemoryState {
	NotInitialized,
	InitializedOnly,
	OnHost,
	OnDevice,
	Crashed
};

inline std::ostream& operator<<(std::ostream& so, GpuMemoryState gState) {
	switch (gState) {
		case NotInitialized:
			so << "NotInitialized";
			break;
		case InitializedOnly:
			so << "InitializedOnly";
			break;
		case OnHost:
			so << "OnHost";
			break;
		case OnDevice:
			so << "OnDevice";
			break;
		case Crashed:
			so << "Crashed";
			break;
		default:
			so.setstate(std::ios_base::failbit);
	}
	return so;
}

template <typename T>
class GNAcuOscProbMem {
       public:
	T* devEnu;
	T* devTmp;
	T* devComp0;
	T* devComp12;
	T* devComp13;
	T* devComp23;
	T* devCompCP;
	T* devRet;
	GpuMemoryState currentGpuMemState;

	GNAcuOscProbMem(int numOfElem);
	~GNAcuOscProbMem();
};

#endif /* GNACUOSCPROBMEM_H */
