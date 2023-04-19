#include "ParCovMatrix.hh"

using GNA::MatrixFormat;

ParCovMatrix::ParCovMatrix(MatrixFormat matrix_format) :
m_permit_diagonal{matrix_format==MatrixFormat::PermitDiagonal}
{
    transformation_("unc_matrix")
        .output("unc_matrix")
        .types(&ParCovMatrix::Types)
        .func(&ParCovMatrix::FillMatrix)
        .finalize();
}

void ParCovMatrix::Types(TypesFunctionArgs fargs) {
    auto& ret=fargs.rets[0];

    if(m_permit_diagonal && m_covariance_diagonal){
        ret = DataType().points().shape(m_pars.size());
    }
    else{
        ret = DataType().points().shape(m_pars.size(), m_pars.size());
    }
}

void ParCovMatrix::append(GaussianParameter<double>* par) {
    if(m_covariance_diagonal){
        for(auto& otherpar: m_pars){
            if(otherpar->isCorrelated(*par)){
                m_covariance_diagonal=false;
                break;
            }
        }
    }

    m_pars.push_back(par);
}

void ParCovMatrix::FillMatrix(FunctionArgs fargs) {
    if(m_covariance_diagonal){
        if (m_permit_diagonal){
            auto& covmat_diag = fargs.rets[0].x;
            for (size_t row{0}; row < m_pars.size(); ++row) {
                covmat_diag(row) = pow(m_pars[row]->sigma(), 2);
            }
        }
        else{
            auto& covmat = fargs.rets[0].arr2d;
            covmat.setZero();
            for (size_t row{0}; row < m_pars.size(); ++row) {
                covmat(row, row) = pow(m_pars[row]->sigma(), 2);
            }
        }

        return;
    }

    auto& covmat = fargs.rets[0].arr2d;
    for (size_t row{0}; row < m_pars.size(); ++row) {
        auto* par_row = m_pars.at(row);
        for (size_t col{0}; col < row; ++col) {
            auto* par_col = m_pars.at(col);
            covmat(row, col) = covmat(col, row) = par_row->getCovariance(*par_col);
        }
        covmat(row, row) = pow(par_row->sigma(), 2);
    }
}

void ParCovMatrix::materialize() {
    t_["unc_matrix"].updateTypes();
}
