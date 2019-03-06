#pragma once

#include "GNAObject.hh"
#include "TypesFunctions.hh"

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
    transformation_("cholesky")
      .input("mat")
      .output("L")
      .types(TypesFunctions::ifSquare<0>, TypesFunctions::pass<0>, &Cholesky::prepareCholesky)
      .func(&Cholesky::calculateCholesky)
    ;
  }
  inline void prepareCholesky(TypesFunctionArgs& fargs);
  inline void calculateCholesky(FunctionArgs& fargs);

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
void Cholesky::prepareCholesky(TypesFunctionArgs& fargs) {
  m_llt = LLT(fargs.args[0].shape[0]);
  fargs.rets[0].preallocated(const_cast<double*>(m_llt.matrixRef().data()));
}

/**
 * Decompose
 */
void Cholesky::calculateCholesky(FunctionArgs& fargs) {
  m_llt.compute(fargs.args[0].mat);
}
