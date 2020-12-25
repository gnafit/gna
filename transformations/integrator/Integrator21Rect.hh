#pragma once

#include "GNAObject.hh"
#include "Integrator21Base.hh"

class Integrator21Rect: public TransformationBind<Integrator21Rect>,
                        public Integrator21Base {
public:
  Integrator21Rect(size_t xbins, int  xorders, double* edges, int yorder, double ymin, double ymax, const std::string& mode="center");
  Integrator21Rect(size_t xbins, int* xorders, double* edges, int yorder, double ymin, double ymax, const std::string& mode="center");

  void sample(FunctionArgs& fargs) final;

protected:
  std::string m_mode;
  int m_rect_offset{0};
};
