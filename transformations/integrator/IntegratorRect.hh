#pragma once

#include "GNAObject.hh"
#include "IntegratorBase.hh"

class IntegratorRect: public TransformationBind<IntegratorRect>,
                      public IntegratorBase {
public:
  IntegratorRect(size_t bins, int orders, double* edges=nullptr, const std::string& mode="center");
  IntegratorRect(size_t bins, int* orders, double* edges=nullptr, const std::string& mode="center");

  void sample(FunctionArgs& fargs) final;

protected:
  void init(const std::string& mode);

  std::string m_mode;
  int m_rect_offset{0};
};

