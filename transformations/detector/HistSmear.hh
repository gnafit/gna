#pragma once

#include "GNAObject.hh"
#include "TypesFunctions.hh"

class HistSmear: public GNASingleObject,
                 public TransformationBind<HistSmear> {
public:
  HistSmear( bool upper=false );

private:
  void calcSmear(FunctionArgs fargs);
  void calcSmearUpper(FunctionArgs fargs);

  void calcSmear_gpu(FunctionArgs &fargs);
  void calcSmearUpper_gpu(FunctionArgs  &fargs);
};
