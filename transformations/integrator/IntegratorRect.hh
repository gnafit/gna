#pragma once

#include "GNAObject.hh"
#include "IntegratorBase.hh"

class IntegratorRect: public TransformationBind<IntegratorRect>,
                 public IntegratorBase {
public:
  using TransformationBind<IntegratorRect>::transformation_;

  IntegratorRect(size_t bins, int orders, double* edges=0, const std::string& mode="center");
  IntegratorRect(size_t bins, int* orders, double* edges=0, const std::string& mode="center");

protected:
  void init(const std::string& mode);
  void check(TypesFunctionArgs& fargs);

  void compute_rect(FunctionArgs& fargs);

  std::string m_mode;
  int m_rect_offset{0};
};

