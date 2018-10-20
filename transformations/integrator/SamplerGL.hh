#pragma once

#include "GNAObject.hh"
#include "SamplerBase.hh"

class SamplerGL: public TransformationBind<SamplerGL>,
                 public SamplerBase {
public:
  using TransformationBind<SamplerGL>::transformation_;

  SamplerGL(size_t bins, int orders, double* edges=0, const std::string& mode="gl");
  SamplerGL(size_t bins, int* orders, double* edges=0, const std::string& mode="gl");

protected:
  void init(const std::string& mode);
  void check(TypesFunctionArgs& fargs);
  void compute_gl(FunctionArgs& fargs);
  void compute_rect(FunctionArgs& fargs);

  std::string m_mode;
  int m_rect_offset{0};
};

