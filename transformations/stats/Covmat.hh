#pragma once

#include "GNAObject.hh"

class Covmat: public GNAObject,
              public TransformationBind<Covmat> {
public:
  Covmat()
    : m_fixed(false)
  {
    transformation_(this, "cov")
      .types(Atypes::ifSameShape, [](Atypes args, Rtypes rets) {
          rets[0] = DataType().points().shape(args[0].size(), args[0].size());
        })
      .func(&Covmat::calculateCov)
      .input("stat")
      .output("cov")
    ;
    transformation_(this, "inv")
      .types(Atypes::pass<0>)
      .func(&Covmat::calculateInv)
      .input("cov")
      .output("inv")
      ;
    transformation_(this, "cholesky")
      .types(Atypes::pass<0>, &Covmat::prepareCholesky)
      .func(&Covmat::calculateCholesky)
      .input("cov")
      .output("L")
    ;
  }
  void calculateCov(Args args, Rets rets);
  void calculateInv(Args args, Rets rets);

  void prepareCholesky(Atypes args, Rtypes rets);
  void calculateCholesky(Args args, Rets rets);

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
