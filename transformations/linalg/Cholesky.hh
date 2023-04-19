#pragma once

#include "GNAObject.hh"
#include "TypesFunctions.hh"
#include "DataEnums.hh"

/**
 * Cholesky tranformation
 *
 * Defines the Cholesky decomposition L of the symmetric positive definite matrix V:
 * V = L L^T,
 * where L is the lower triangular matrix. 1d input may optionally be treated as a diagonal.
 */
class Cholesky: public GNASingleObject,
public TransformationBind<Cholesky> {
public:
    Cholesky(GNA::MatrixFormat matrix_format=GNA::MatrixFormat::Regular);

    void prepareCholesky(TypesFunctionArgs& fargs);
    void calculateCholesky(FunctionArgs& fargs);

protected:
    class LLT: public Eigen::LLT<Eigen::MatrixXd> {

public:
        LLT(): Eigen::LLT<Eigen::MatrixXd>() { }
        LLT(size_t size): Eigen::LLT<Eigen::MatrixXd>(size) { }
        Eigen::MatrixXd &matrixRef() { return this->m_matrix; }
    };

    std::unique_ptr<LLT> m_llt;
    bool m_permit_diagonal = false;
};
