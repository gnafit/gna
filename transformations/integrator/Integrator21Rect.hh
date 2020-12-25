#pragma once

#include "GNAObject.hh"
#include "Integrator21Base.hh"

class Integrator21Rect: public TransformationBind<Integrator21Rect>,
                        public Integrator21Base {
public:
  Integrator21Rect(size_t xbins, int  xorders, double* edges, int yorder, double ymin, double ymax);
  Integrator21Rect(size_t xbins, int* xorders, double* edges, int yorder, double ymin, double ymax);

  void sample(FunctionArgs& fargs) final;
};

