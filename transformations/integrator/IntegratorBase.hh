#pragma once

#include "GNAObject.hh"
#include "GNAObjectBind1N.hh"
#include <Eigen/Dense>
#include <boost/optional.hpp>

class IntegratorBase : public GNAObjectBind1N<double>,
                       public TransformationBind<IntegratorBase>
{
public:
  void set_edges(OutputDescriptor& hist);
  TransformationDescriptor add_transformation(const std::string& name="");

  void dump();

  const Eigen::ArrayXd& getWeights() { return m_weights; }
  const Eigen::ArrayXi& getOrders() { return m_orders; }
protected:
  IntegratorBase(int order, bool shared_edge=false);
  IntegratorBase(size_t bins, int orders, double* edges=nullptr, bool shared_edge=false);
  IntegratorBase(size_t bins, int* orders, double* edges=nullptr, bool shared_edge=false);

  Eigen::ArrayXi m_orders;
  Eigen::ArrayXd m_weights;
  Eigen::ArrayXd m_edges;

  boost::optional<size_t> m_order;

  virtual void sample(FunctionArgs&)=0;

protected:
  void init_sampler();
  void check_sampler(TypesFunctionArgs& fargs);

  size_t shared_edge() { return m_shared_edge; }

private:
  void init_base(double* edges);
  void check_base(TypesFunctionArgs&);
  void integrate(FunctionArgs&);

  size_t m_shared_edge{0};
};
