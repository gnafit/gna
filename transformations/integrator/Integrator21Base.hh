#pragma once

#include "GNAObject.hh"
#include <Eigen/Dense>

class Integrator21Base : public GNAObject,
                       public TransformationBind<Integrator21Base>
{
public:
  void dump();

protected:
  Integrator21Base(size_t xbins, int  xorders, double* edges, int yorder, double ymin, double ymax);
  Integrator21Base(size_t xbins, int* xorders, double* edges, int yorder, double ymin, double ymax);

  Eigen::ArrayXi m_xorders;
  Eigen::ArrayXd m_xedges;
  Eigen::ArrayXd m_xweights;
  Eigen::ArrayXd m_yweights;

  int m_yorder;
  double ymin;
  double ymax;

  virtual void sample(FunctionArgs&)=0;

protected:
  void init_sampler();
  void check_sampler(TypesFunctionArgs& fargs);

private:
  void init_base(double* xedges);
  void check_base(TypesFunctionArgs&);
  void integrate(FunctionArgs&);
};
