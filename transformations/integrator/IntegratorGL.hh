#pragma once

#include "GNAObject.hh"
#include "IntegratorBase.hh"

class IntegratorGL: public TransformationBind<IntegratorGL>,
                    public IntegratorBase {
public:
  IntegratorGL(int orders);
  IntegratorGL(size_t bins, int orders, double* edges=nullptr);
  IntegratorGL(size_t bins, int* orders, double* edges=nullptr);

  void sample(FunctionArgs& fargs) final;
};

