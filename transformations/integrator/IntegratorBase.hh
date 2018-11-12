#pragma once

#include "GNAObject.hh"
#include <Eigen/Dense>

class IntegratorBase : public GNAObject,
                       public TransformationBind<IntegratorBase>
{
public:
  TransformationDescriptor add();

  void dump();

protected:
  IntegratorBase(size_t bins, int orders, double* edges=nullptr, bool shared_edge=false);
  IntegratorBase(size_t bins, int* orders, double* edges=nullptr, bool shared_edge=false);

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
