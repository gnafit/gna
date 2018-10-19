#pragma once

#include "GNAObject.hh"
#include <Eigen/Dense>

class SamplerBase : public GNAObject,
                    public TransformationBind<SamplerBase>
{
public:
  void dump();

protected:
  SamplerBase(size_t bins, int orders, double* edges=0);
  SamplerBase(size_t bins, int* orders, double* edges=0);

  Eigen::ArrayXi m_orders;
  Eigen::ArrayXd m_weights;
  Eigen::ArrayXd m_edges;

private:
  void init_base(double* edges);
  void check_base(TypesFunctionArgs&);
  void integrate(FunctionArgs&);
};
