#pragma once

#include "GNAObject.hh"
#include "GNAObjectBind1N.hh"
#include <Eigen/Dense>

class Integrator2Base : public GNAObjectBind1N<double>,
                        public TransformationBind<Integrator2Base>
{
public:
  TransformationDescriptor add_transformation(const std::string& name="");
  void set_edges(OutputDescriptor& hist2_output);

  void dump();

protected:
  Integrator2Base(size_t xbins, int  xorders, double* xedges, size_t ybins, int  yorders, double* yedges);
  Integrator2Base(size_t xbins, int* xorders, double* xedges, size_t ybins, int* yorders, double* yedges);

  Eigen::ArrayXi m_xorders;
  Eigen::ArrayXd m_xedges;
  Eigen::ArrayXd m_xweights;

  Eigen::ArrayXi m_yorders;
  Eigen::ArrayXd m_yedges;
  Eigen::ArrayXd m_yweights;

  Eigen::ArrayXXd m_weights;

  virtual void sample(FunctionArgs&)=0;

protected:
  void init_sampler();
  void check_sampler(TypesFunctionArgs& fargs);

private:
  void init_base(double* xedges, double* yedges);
  void check_base(TypesFunctionArgs&);
  void integrate(FunctionArgs&);
};
