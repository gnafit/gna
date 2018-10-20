#pragma once

#include "GNAObject.hh"
#include "IntegratorBase.hh"

class IntegratorGL: public TransformationBind<IntegratorGL>,
                    public IntegratorBase {
public:
  IntegratorGL(size_t bins, int orders, double* edges=0);
  IntegratorGL(size_t bins, int* orders, double* edges=0);

  void sample(FunctionArgs& fargs) final;
};

