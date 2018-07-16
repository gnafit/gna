#pragma once

#include "GNAObject.hh"
#include "TypesFunctions.hh"

class Covmat: public GNAObject,
              public TransformationBind<Covmat> {
public:
  Covmat()
    : m_fixed(false)
  {
    transformation_("cov")
      .input("stat")
      .output("cov")
      .types(TypesFunctions::ifSameShape, TypesFunctions::toMatrix<0,0,0>)
      .func(&Covmat::calculateCov)
    ;
    transformation_("inv")
      .input("cov")
      .output("inv")
      .types(TypesFunctions::ifSquare<0>, TypesFunctions::pass<0>)
      .func(&Covmat::calculateInv)
      ;
    transformation_("cholesky")
      .types(TypesFunctions::ifSquare<0>, TypesFunctions::pass<0>, &Covmat::prepareCholesky)
      .func(&Covmat::calculateCholesky)
      .input("cov")
      .output("L")
    ;
  }
  void calculateCov(FunctionArgs fargs);
  void calculateInv(FunctionArgs fargs);

  void prepareCholesky(TypesFunctionArgs fargs);
  void calculateCholesky(FunctionArgs fargs);

  void rank1(SingleOutput &data);

  size_t ndim() const;
  size_t size() const;

  void update() const;
  const double *data() const;

  void setFixed(bool fixed) { m_fixed = fixed; }
protected:
  class LLT: public Eigen::LLT<Eigen::MatrixXd> {
  public:
    LLT(): Eigen::LLT<Eigen::MatrixXd>() { }
    LLT(size_t size): Eigen::LLT<Eigen::MatrixXd>(size) { }
    Eigen::MatrixXd &matrixRef() { return this->m_matrix; }
  };

  bool m_fixed;
  LLT m_llt;
};
