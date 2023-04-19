#pragma once

#include "GNAObject.hh"
#include "config_vars.h"

#include "HistSmearEnums.hh"

class HistSmear: public GNASingleObject,
                 public TransformationBind<HistSmear> {
public:
  HistSmear(GNA::SquareMatrixType matrix_type=GNA::SquareMatrixType::Any);

private:
  void calcSmear(FunctionArgs fargs);
  void calcSmearUpper(FunctionArgs fargs);

#ifdef GNA_CUDA_SUPPORT
  void calcSmear_gpu(FunctionArgs &fargs);
  void calcSmearUpper_gpu(FunctionArgs  &fargs);
#endif
};
