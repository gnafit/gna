#pragma once

#include "GNAObject.hh"
#include "IntegratorBase.hh"

class IntegratorGL: public TransformationBind<IntegratorGL>,
                    public IntegratorBase {
public:
  using TransformationBind<IntegratorGL>::transformation_;

  IntegratorGL(size_t bins, int orders, double* edges=0);
  IntegratorGL(size_t bins, int* orders, double* edges=0);

protected:
  void init();
  void check(TypesFunctionArgs& fargs);

  void compute_gl(FunctionArgs& fargs);
};

