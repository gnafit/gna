#pragma once

#include "GNAObject.hh"

class HistSmear: public GNASingleObject,
                 public TransformationBind<HistSmear> {
public:
  HistSmear( bool upper=false );

private:
  void calcSmear(Args args, Rets rets);
  void calcSmearUpper(Args args, Rets rets);
};
