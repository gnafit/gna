#pragma once

#include "GNAObject.hh"
#include <Eigen/Dense>

class Integrator21Base : public GNAObject,
                       public TransformationBind<Integrator21Base>
{
public:
  TransformationDescriptor add();

  void dump();

protected:
  Integrator21Base(size_t xbins, int  xorders, double* edges, int yorder, double ymin, double ymax);
  Integrator21Base(size_t xbins, int* xorders, double* edges, int yorder, double ymin, double ymax);

  Eigen::ArrayXi m_xorders;
  Eigen::ArrayXd m_xedges;
  Eigen::ArrayXd m_xweights;

  int m_yorder;
  double m_ymin;
  double m_ymax;
  Eigen::ArrayXd m_yweights;

  Eigen::ArrayXXd m_weights;

  virtual void sample(FunctionArgs&)=0;

protected:
  void init_sampler();
  void check_sampler(TypesFunctionArgs& fargs);

private:
  void init_base(double* xedges);
  void check_base(TypesFunctionArgs&);
  void integrate(FunctionArgs&);
};
