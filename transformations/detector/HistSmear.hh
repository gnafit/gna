#pragma once

#include "GNAObject.hh"

class HistSmear: public GNASingleObject,
                 public TransformationBind<HistSmear> {
public:
  HistSmear( bool upper=false );

private:
  void calcSmear(FunctionArgs fargs);
  void calcSmearUpper(FunctionArgs fargs);
};
