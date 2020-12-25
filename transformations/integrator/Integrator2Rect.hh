#pragma once

#include "GNAObject.hh"
#include "Integrator2Base.hh"


class Integrator2Rect: public TransformationBind<Integrator2Rect>,
                       public Integrator2Base {
public:
  Integrator2Rect(size_t xbins, int  xorders, size_t ybins, int  yorders, const std::string& mode="center");
  Integrator2Rect(size_t xbins, int* xorders, size_t ybins, int* yorders, const std::string& mode="center");
  Integrator2Rect(size_t xbins, int  xorders, double* xedges, size_t ybins, int  yorders, double* yedges, const std::string& mode="center");
  Integrator2Rect(size_t xbins, int* xorders, double* xedges, size_t ybins, int* yorders, double* yedges, const std::string& mode="center");

  void sample(FunctionArgs& fargs) final;

protected:
  void init(const std::string& mode);
  void calcWeights(int rect_offset, size_t nbins, int* order, double* edges, double* abscissa, double* weight);

  std::string m_mode;
  int m_rect_offset{0};
};
