#pragma once

#include "GNAObject.hh"
#include <Eigen/Dense>

class IntegratorBase : public GNAObject,
                       public TransformationBind<IntegratorBase>
{
public:
  TransformationDescriptor add_transformation();

  void set_edges(OutputDescriptor& hist);
  InputDescriptor  add_input();
  OutputDescriptor add_input(OutputDescriptor& fcn_output);
  OutputDescriptor add_input(InputDescriptor& fcn_input, OutputDescriptor& fcn_output);
  OutputDescriptor add_input(OutputDescriptor& hist, InputDescriptor& fcn_input, OutputDescriptor& fcn_output);

  void dump();

protected:
  IntegratorBase(size_t bins, int orders, double* edges=0, bool shared_edge=false);
  IntegratorBase(size_t bins, int* orders, double* edges=0, bool shared_edge=false);

  Eigen::ArrayXi m_orders;
  Eigen::ArrayXd m_weights;
  Eigen::ArrayXd m_edges;

  virtual void sample(FunctionArgs&)=0;

protected:
  void init_sampler();
  void check_sampler(TypesFunctionArgs& fargs);

private:
  void init_base(double* edges);
  void check_base(TypesFunctionArgs&);
  void integrate(FunctionArgs&);

  size_t m_shared_edge{0};
};
