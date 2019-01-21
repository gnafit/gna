#pragma once

#include "GNAObject.hh"
#include "IntegratorBase.hh"

class IntegratorTrap: public TransformationBind<IntegratorTrap>,
                      public IntegratorBase {
public:
  IntegratorTrap(size_t bins, int orders, double* edges=nullptr);
  IntegratorTrap(size_t bins, int* orders, double* edges=nullptr);

  void sample(FunctionArgs& fargs) final;
};

