#pragma once

#include <Eigen/Dense>
#include "GNAObject.hh"

class SamplerGL: public GNAObject,
                 public TransformationBind<SamplerGL> {
public:
  SamplerGL(size_t bins, int orders)        : m_orders(bins)  { m_orders=orders; init(); }
  SamplerGL(size_t bins, const int *orders) : m_orders(Eigen::Map<const Eigen::ArrayXi>(orders, bins)) { init(); }
  SamplerGL(const std::vector<int> &orders) : m_orders(Eigen::Map<const Eigen::ArrayXi>(orders.data(), orders.size())) { init(); }

protected:
  void init();
  void check(TypesFunctionArgs& fargs);
  void compute(FunctionArgs& fargs);
  Eigen::ArrayXi m_orders;
};

