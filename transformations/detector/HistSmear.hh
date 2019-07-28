#pragma once

#include "GNAObject.hh"
#include "config_vars.h"

class HistSmear: public GNASingleObject,
                 public TransformationBind<HistSmear> {
public:
  HistSmear( bool upper=false );

private:
  void calcSmear(FunctionArgs fargs);
  void calcSmearUpper(FunctionArgs fargs);

#ifdef GNA_CUDA_SUPPORT
  void calcSmear_gpu(FunctionArgs &fargs);
  void calcSmearUpper_gpu(FunctionArgs  &fargs);
#endif
};
