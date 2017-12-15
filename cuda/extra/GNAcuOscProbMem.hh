#ifndef GNACUOSCPROBMEM_H
#define GNACUOSCPROBMEM_H

#include <iostream>

template<typename T>
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

  GNAcuOscProbMem(int numOfElem);
  ~GNAcuOscProbMem();
};

#endif /* GNACUOSCPROBMEM_H */

