#include "ParCovMatrix.hh"
#include "fmt/ostream.h"

void ParCovMatrix::FillMatrix(FunctionArgs fargs) {
    Eigen::MatrixXd pars_covmat{m_pars.size(), m_pars.size()};
    pars_covmat.setZero();
    for (size_t row{0}; row < m_pars.size(); ++row) {
        for (size_t col{row}; col < m_pars.size(); ++col) {
            auto*  primary_par = m_pars.at(row);
            auto*  secondary_par = m_pars.at(col);
            pars_covmat(row, col) = pars_covmat(col, row) = primary_par->getCovariance(*secondary_par);
        }
    }
    fargs.rets[0].mat = pars_covmat;
}

void ParCovMatrix::materialize() {
    t_["unc_matrix"].updateTypes();
}
