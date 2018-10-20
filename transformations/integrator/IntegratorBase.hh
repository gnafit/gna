#pragma once

#include "GNAObject.hh"
#include <Eigen/Dense>

class IntegratorBase : public GNAObject,
                    public TransformationBind<IntegratorBase>
{
public:
  void dump();

protected:
  IntegratorBase(size_t bins, int orders, double* edges=0);
  IntegratorBase(size_t bins, int* orders, double* edges=0);

  Eigen::ArrayXi m_orders;
  Eigen::ArrayXd m_weights;
  Eigen::ArrayXd m_edges;

  void set_shared_edge();

private:
  void init_base(double* edges);
  void check_base(TypesFunctionArgs&);
  void integrate(FunctionArgs&);

  size_t m_shared_edge{0};
};
