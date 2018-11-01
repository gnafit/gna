#pragma once

#include "GNAObject.hh"
#include "Integrator21Base.hh"

class Integrator21GL: public TransformationBind<Integrator21GL>,
                      public Integrator21Base {
public:
  Integrator21GL(size_t xbins, int  xorders, double* edges, int yorder, double ymin, double ymax);
  Integrator21GL(size_t xbins, int* xorders, double* edges, int yorder, double ymin, double ymax);

  void sample(FunctionArgs& fargs) final;
};

