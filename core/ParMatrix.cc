#include "ParMatrix.hh"
#include "boost/format.hpp"
using boost::format;
void ParMatrix::FillMatrix(Args args, Rets rets) {
    Eigen::MatrixXd pars_covmat{m_pars.size(), m_pars.size()};
    pars_covmat.setZero();
    std::cout << "pars_covmat.size() = " << pars_covmat.size() << std::endl;
    for (size_t row{0}; row < m_pars.size(); ++row) {
        for (size_t col{row}; col < m_pars.size(); ++col) {
            auto*  primary_par __attribute__((unused))= m_pars.at(row);
            auto*  secondary_par __attribute__((unused)) = m_pars.at(col);
            std::cout << format("row %1%, col %2%") % row % col << std::endl;

            pars_covmat(row, col) = primary_par->getCovariance(*secondary_par);
        }
    }
        std::cout << "Before symmetrizing: " << std::endl;
        std::cout << pars_covmat << std::endl;
        pars_covmat.matrix().triangularView<Eigen::StrictlyLower>() = pars_covmat.matrix().triangularView<Eigen::StrictlyUpper>();
        std::cout << "After symmetrizing: " << std::endl;
        /* std::cout << pars_covmat << std::endl; */
        rets[0].mat = pars_covmat;
        std::cout << rets[0].mat << std::endl;

}

void ParMatrix::materialize() {
    t_["unc_matrix"].updateTypes();
}
