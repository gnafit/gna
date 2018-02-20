#ifndef CHOLESKY_H
#define CHOLESKY_H

#include "GNAObject.hh"

/**
 * Cholesky tranformation
 *
 * Defines the Cholesky decomposition L of the symmetric positive definite matrix V:
 * V = L L^T,
 * where L is the lower triangular matrix.
 */
class Cholesky: public GNASingleObject,
                public TransformationBind<Cholesky> {
public:
  Cholesky()
  {
    transformation_(this, "cholesky")
      .input("mat")
      .output("L")
      .types(Atypes::pass<0>, &Cholesky::prepareCholesky)
      .func(&Cholesky::calculateCholesky)
    ;
  }
  void prepareCholesky(Atypes args, Rtypes rets);
  void calculateCholesky(Args args, Rets rets);

protected:
  class LLT: public Eigen::LLT<Eigen::MatrixXd> {
  public:
    LLT(): Eigen::LLT<Eigen::MatrixXd>() { }
    LLT(size_t size): Eigen::LLT<Eigen::MatrixXd>(size) { }
    Eigen::MatrixXd &matrixRef() { return this->m_matrix; }
  };

  LLT m_llt;
};

/**
 * Check that the input is matrix and the matrix is symmetric
 */
void Cholesky::prepareCholesky(Atypes args, Rtypes rets) {
  if (args[0].shape.size() != 2) {
    throw args.error(args[0], "Cholesky decomposition can't be performed on non-2d data");
  }
  if (args[0].shape[0] != args[0].shape[1]) {
    throw args.error(args[0], "Cholesky decomposition can't be performed with non-square matrix");
  }
  m_llt = LLT(args[0].shape[0]);
  rets[0].preallocated(const_cast<double*>(m_llt.matrixRef().data()));
}

/**
 * Decompose
 */
void Cholesky::calculateCholesky(Args args, Rets rets) {
  m_llt.compute(args[0].mat);
}

#endif // CHOLESKY_H
