#pragma once

#include <vector>

#include "GNAObject.hh"
#include "Eigen/Sparse"

class RebinN: public GNAObject,
              public TransformationBind<RebinN> {
public:
  RebinN(size_t n);

  void doRebin(FunctionArgs& fargs);
  void doTypes(TypesFunctionArgs& fargs);
private:
  size_t m_njoin;
};
