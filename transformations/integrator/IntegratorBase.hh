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
  IntegratorBase(int order);
  IntegratorBase(size_t bins, int orders, double* edges=nullptr);
  IntegratorBase(size_t bins, int* orders, double* edges=nullptr);

  Eigen::ArrayXi m_orders;
  Eigen::ArrayXd m_weights;
  Eigen::ArrayXd m_edges;

  boost::optional<size_t> m_order;

  virtual void sample(FunctionArgs&)=0;

protected:
  void init_sampler();
  void check_sampler(TypesFunctionArgs& fargs);

private:
  void init_base(double* edges);
  void check_base(TypesFunctionArgs&);
  void integrate(FunctionArgs&);
};
