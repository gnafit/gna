#pragma once

#include "GNAObject.hh"
#include "Integrator2Base.hh"

class Integrator2GL: public TransformationBind<Integrator2GL>,
                     public Integrator2Base {
public:
  Integrator2GL(size_t xbins, int  xorders, size_t ybins, int  yorders);
  Integrator2GL(size_t xbins, int* xorders, size_t ybins, int* yorders);
  Integrator2GL(size_t xbins, int  xorders, double* edges, size_t ybins, int  yorders, double* yedges);
  Integrator2GL(size_t xbins, int* xorders, double* edges, size_t ybins, int* yorders, double* yedges);

  void sample(FunctionArgs& fargs) final;
};

