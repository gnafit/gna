#pragma once

#include "GNAObject.hh"
#include "IntegratorBase.hh"

class IntegratorTrap: public TransformationBind<IntegratorTrap>,
                      public IntegratorBase {
public:
  using TransformationBind<IntegratorTrap>::transformation_;

  IntegratorTrap(size_t bins, int orders, double* edges=0);
  IntegratorTrap(size_t bins, int* orders, double* edges=0);

protected:
  void init();
  void check(TypesFunctionArgs& fargs);

  void compute_trap(FunctionArgs& fargs);
};

