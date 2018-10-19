#pragma once

#include "GNAObject.hh"
#include "SamplerBase.hh"

class SamplerGL: public TransformationBind<SamplerGL>,
                 public SamplerBase {
public:
  using TransformationBind<SamplerGL>::transformation_;

  SamplerGL(size_t bins, int orders, double* edges=0);
  SamplerGL(size_t bins, int* orders, double* edges=0);

protected:
  void init();
  void check(TypesFunctionArgs& fargs);
  void compute(FunctionArgs& fargs);
};

