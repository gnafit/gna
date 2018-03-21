#include "ParMatrix.hh"


void ParMatrix::FillMatrix(Args args, Rets rets) {
    for (size_t row{0}; row < m_pars.size(); ++row) {
        for (size_t col{row}; col < m_pars.size(); ++col) {
            auto* primary_par = m_pars.at(row);
            auto* secondary_par = m_pars.at(col);
            if (row!=col) { assert(primary_par != secondary_par); };
            rets[0].mat(row, col) = primary_par->getCovariance(*secondary_par);
        }
        std::cout << rets[0].mat << std::endl;
    }
}
